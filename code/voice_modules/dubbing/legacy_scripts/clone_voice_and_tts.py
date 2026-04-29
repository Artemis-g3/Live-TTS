from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import winsound

import dashscope
import numpy as np
import requests
import soundfile as sf
import torch
from dashscope import MultiModalConversation
from voxcpm import VoxCPM

from bailian_retrieval_common import (
    COL_TRANSCRIPT,
    DEFAULT_FINAL_RANKING_PROMPT_PATH,
    ensure_out_dir,
    get_api_key,
)
from role_data import RolePaths, ensure_role_excel
from search_double_keyword_retrieval import print_retrieval_result, run_double_keyword_retrieval

VOICE_ENROLLMENT_MODEL = "qwen-voice-enrollment"
VOICE_ENROLLMENT_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
DEFAULT_TARGET_MODEL = "qwen3-tts-vc-2026-01-22"
DEFAULT_BACKEND = "api"
BACKEND_CHOICES = ("api", "voxcpm2_local_basic", "voxcpm2_local_hifi")
DEFAULT_REFERENCE_WINDOW = 10
DEFAULT_REFERENCE_MIN_SECONDS = 15.0
DEFAULT_GAP_MS = 300
DEFAULT_REFERENCE_LANGUAGE = "zh"
DEFAULT_TTS_LANGUAGE_TYPE = "Chinese"
LOCAL_VOXCPM2_MODEL_PATH = Path(__file__).resolve().parents[2] / "VoxCPM2"
LOCAL_VOXCPM2_INFERENCE_TIMESTEPS = 20
LOCAL_VOXCPM2_CFG_VALUE = 2.0
FFPLAY_CANDIDATES = (
    Path(r"C:\Program Files\ffmpeg\bin\ffplay.exe"),
    Path("ffplay.exe"),
)

_LOCAL_VOXCPM2_MODEL: VoxCPM | None = None


def read_required_text(arg_text: str | None, prompt_text: str, empty_message: str) -> str:
    if arg_text and arg_text.strip():
        return arg_text.strip()
    text = input(prompt_text).strip()
    if not text:
        raise ValueError(empty_message)
    return text


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "items") and not isinstance(value, (str, bytes)):
        try:
            return {str(key): to_jsonable(item) for key, item in dict(value).items()}
        except Exception:
            pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def make_role_output_paths(role_paths: RolePaths) -> dict[str, Path]:
    role_root = role_paths.output_dir.resolve()
    return {
        "role_root": role_root,
        "tts_runs_dir": role_root / "tts_runs",
        "voice_cache_path": role_root / "voice_cache.json",
        "voxcpm2_prompt_cache_index_path": role_root / "voxcpm2_prompt_cache.json",
        "voxcpm2_prompts_dir": role_root / "voxcpm2_prompt_cache",
    }


def make_session_dir(base_dir: Path) -> Path:
    base_dir = base_dir.resolve()
    ensure_out_dir(base_dir)
    session_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = 1
    while session_dir.exists():
        session_dir = base_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix:02d}"
        suffix += 1
    ensure_out_dir(session_dir)
    return session_dir


def load_cache_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"fingerprints": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"fingerprints": {}}
    if not isinstance(data, dict):
        return {"fingerprints": {}}
    fingerprints = data.get("fingerprints", {})
    if not isinstance(fingerprints, dict):
        fingerprints = {}
    return {"fingerprints": fingerprints}


def save_cache_index(cache: dict[str, Any], path: Path) -> None:
    ensure_out_dir(path.parent)
    write_json(path, cache)


def build_reference_audio(
    top_results: list[dict[str, Any]],
    *,
    reference_window: int,
    gap_ms: int,
    reference_min_seconds: float,
    session_dir: Path,
) -> dict[str, Any]:
    selected_rows = top_results[:reference_window]
    if not selected_rows:
        raise RuntimeError("没有可用于声音复刻的检索结果。")

    combined_segments: list[np.ndarray] = []
    selected_clips: list[dict[str, Any]] = []
    reference_texts: list[str] = []
    sample_rate: int | None = None
    total_samples = 0

    for row in selected_rows:
        audio_path = Path(str(row["audio_path"]))
        if not audio_path.exists():
            raise FileNotFoundError(f"参考音频文件不存在: role={row.get('role', '')} path={audio_path}")

        audio_data, clip_sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        clip_duration_seconds = len(audio_data) / float(clip_sample_rate)
        if clip_duration_seconds > reference_min_seconds:
            continue

        if sample_rate is None:
            sample_rate = int(clip_sample_rate)
        elif int(clip_sample_rate) != sample_rate:
            raise RuntimeError(
                f"参考音频采样率不一致: {audio_path.name}={clip_sample_rate}, expected={sample_rate}"
            )

        if combined_segments:
            gap = np.zeros(int(round(sample_rate * (gap_ms / 1000.0))), dtype=np.float32)
            combined_segments.append(gap)
            total_samples += len(gap)

        combined_segments.append(audio_data.astype(np.float32))
        total_samples += len(audio_data)
        selected_clips.append(row)
        reference_texts.append(str(row[COL_TRANSCRIPT]))

        if sample_rate and (total_samples / sample_rate) > reference_min_seconds:
            break

    if sample_rate is None or not combined_segments:
        raise RuntimeError("未能生成参考音频。")

    reference_audio = np.concatenate(combined_segments).astype(np.float32)
    reference_audio_path = session_dir / "reference_audio.wav"
    sf.write(str(reference_audio_path), reference_audio, sample_rate, subtype="PCM_16")

    reference_text = "\n".join(reference_texts)
    reference_text_path = session_dir / "reference_text.txt"
    reference_text_path.write_text(reference_text, encoding="utf-8")

    duration_seconds = len(reference_audio) / sample_rate
    return {
        "selected_results": selected_clips,
        "reference_audio_path": reference_audio_path,
        "reference_text_path": reference_text_path,
        "reference_text": reference_text,
        "sample_rate": sample_rate,
        "duration_seconds": duration_seconds,
    }


def compute_reference_fingerprint(
    *,
    reference_audio_path: Path,
    reference_text: str,
    target_model: str,
    language: str,
) -> str:
    payload = hashlib.sha256()
    payload.update(reference_audio_path.read_bytes())
    payload.update(reference_text.encode("utf-8"))
    payload.update(target_model.encode("utf-8"))
    payload.update(language.encode("utf-8"))
    return payload.hexdigest()


def compute_local_prompt_fingerprint(
    *,
    reference_audio_path: Path,
    reference_text: str,
    model_path: Path,
    mode: str,
) -> str:
    payload = hashlib.sha256()
    payload.update(reference_audio_path.read_bytes())
    payload.update(reference_text.encode("utf-8"))
    payload.update(str(model_path.resolve()).encode("utf-8"))
    payload.update(mode.encode("utf-8"))
    return payload.hexdigest()


def make_data_url(path: Path, mime_type: str = "audio/wav") -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def create_voice(
    *,
    reference_audio_path: Path,
    reference_text: str,
    target_model: str,
    language: str,
    preferred_name: str,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": VOICE_ENROLLMENT_MODEL,
        "input": {
            "action": "create",
            "target_model": target_model,
            "audio": {"data": make_data_url(reference_audio_path)},
            "text": reference_text,
            "language": language,
            "preferred_name": preferred_name,
        },
    }
    response = requests.post(VOICE_ENROLLMENT_ENDPOINT, headers=headers, json=payload, timeout=300)
    if not response.ok:
        raise RuntimeError(f"声音复刻请求失败: status={response.status_code} body={response.text}")
    data = response.json()
    if data.get("code"):
        raise RuntimeError(f"{data['code']}: {data.get('message', '')}")

    output = data.get("output", {}) or {}
    voice = output.get("voice")
    if isinstance(voice, dict):
        voice = voice.get("name") or voice.get("voice") or voice.get("id")
    if not isinstance(voice, str) or not voice.strip():
        raise RuntimeError(f"声音复刻返回中缺少 voice 字段: {data}")
    return {"voice": voice.strip(), "response": data}


def synthesize_api_tts(*, text: str, voice: str, target_model: str) -> tuple[str, Any]:
    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
    response = MultiModalConversation.call(
        api_key=get_api_key(),
        model=target_model,
        text=text,
        voice=voice,
        language_type=DEFAULT_TTS_LANGUAGE_TYPE,
        stream=False,
    )
    if getattr(response, "status_code", None) != 200:
        raise RuntimeError(
            f"TTS 调用失败: status={getattr(response, 'status_code', None)} "
            f"code={getattr(response, 'code', '')} message={getattr(response, 'message', '')}"
        )

    audio_url = getattr(getattr(response, "output", None), "audio", None)
    if audio_url is not None:
        audio_url = getattr(audio_url, "url", None)
    if not audio_url:
        raise RuntimeError(f"TTS 返回中缺少音频 URL: {response}")
    return str(audio_url), response


def download_file(url: str, output_path: Path) -> str:
    response = requests.get(url, timeout=300)
    if not response.ok:
        raise RuntimeError(f"下载音频失败: status={response.status_code} url={url}")
    output_path.write_bytes(response.content)
    return response.headers.get("Content-Type", "")


def choose_output_audio_path(session_dir: Path, audio_url: str, content_type: str = "") -> Path:
    content_type = content_type.lower()
    if "mpeg" in content_type or "mp3" in content_type:
        return session_dir / "synthesized.mp3"
    if "wav" in content_type or "wave" in content_type:
        return session_dir / "synthesized.wav"

    suffix = Path(urlparse(audio_url).path).suffix.lower()
    if suffix not in {".wav", ".mp3"}:
        suffix = ".wav"
    return session_dir / f"synthesized{suffix}"


def _mci_send(command: str) -> None:
    error_buffer = ctypes.create_unicode_buffer(1024)
    result = ctypes.windll.winmm.mciSendStringW(command, None, 0, 0)
    if result != 0:
        ctypes.windll.winmm.mciGetErrorStringW(result, error_buffer, len(error_buffer))
        raise RuntimeError(error_buffer.value or f"MCI error {result}")


def play_wav(path: Path) -> None:
    winsound.PlaySound(str(path), winsound.SND_FILENAME)


def find_ffplay() -> str:
    for candidate in FFPLAY_CANDIDATES:
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
    return "ffplay.exe"


def play_audio_with_ffplay(path: Path) -> None:
    ffplay = find_ffplay()
    try:
        completed = subprocess.run(
            [ffplay, "-autoexit", "-nodisp", "-loglevel", "error", str(path)],
            check=True,
        )
        if completed.returncode == 0:
            return
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffplay not found: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffplay exited with code {exc.returncode}") from exc


def play_media_with_mci(path: Path) -> None:
    alias = f"codex_{os.getpid()}"
    escaped = str(path).replace("'", "''")
    _mci_send(f"open '{escaped}' alias {alias}")
    try:
        _mci_send(f"play {alias} wait")
    finally:
        try:
            _mci_send(f"close {alias}")
        except Exception:
            pass


def play_audio(path: Path) -> None:
    errors: list[str] = []

    try:
        play_audio_with_ffplay(path)
        return
    except Exception as exc:
        errors.append(f"ffplay={exc}")

    if path.suffix.lower() == ".wav":
        try:
            play_wav(path)
            return
        except Exception as exc:
            errors.append(f"winsound={exc}")

    try:
        play_media_with_mci(path)
        return
    except Exception as exc:
        errors.append(f"mci={exc}")

    raise RuntimeError(f"自动播放失败: {'; '.join(errors)}")


def get_or_create_voice(
    *,
    fingerprint: str,
    reference_audio_path: Path,
    reference_text: str,
    target_model: str,
    language: str,
    cache_path: Path,
    cache: dict[str, Any],
) -> tuple[str, bool, dict[str, Any] | None]:
    cache_entry = cache["fingerprints"].get(fingerprint)
    if isinstance(cache_entry, dict):
        voice = cache_entry.get("voice")
        if isinstance(voice, str) and voice.strip():
            return voice.strip(), True, None

    preferred_name = f"vc_{fingerprint[:10]}"
    created = create_voice(
        reference_audio_path=reference_audio_path,
        reference_text=reference_text,
        target_model=target_model,
        language=language,
        preferred_name=preferred_name,
    )
    cache["fingerprints"][fingerprint] = {
        "voice": created["voice"],
        "preferred_name": preferred_name,
        "target_model": target_model,
        "language": language,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_cache_index(cache, cache_path)
    return created["voice"], False, created["response"]


def remove_cached_voice(fingerprint: str, cache: dict[str, Any], cache_path: Path) -> None:
    if fingerprint in cache["fingerprints"]:
        del cache["fingerprints"][fingerprint]
        save_cache_index(cache, cache_path)


def load_local_voxcpm2_model(model_path: Path = LOCAL_VOXCPM2_MODEL_PATH) -> VoxCPM:
    global _LOCAL_VOXCPM2_MODEL
    if _LOCAL_VOXCPM2_MODEL is not None:
        return _LOCAL_VOXCPM2_MODEL
    if not model_path.exists():
        raise FileNotFoundError(f"Local VoxCPM2 model directory was not found: {model_path}")

    _LOCAL_VOXCPM2_MODEL = VoxCPM.from_pretrained(
        str(model_path),
        load_denoiser=False,
        local_files_only=True,
        optimize=True,
    )
    return _LOCAL_VOXCPM2_MODEL


def create_or_load_voxcpm2_prompt_cache(
    *,
    model: VoxCPM,
    reference_audio_path: Path,
    reference_text: str,
    mode: str,
    cache_index_path: Path,
    prompts_dir: Path,
    model_path: Path = LOCAL_VOXCPM2_MODEL_PATH,
) -> tuple[dict[str, Any], str, bool]:
    fingerprint = compute_local_prompt_fingerprint(
        reference_audio_path=reference_audio_path,
        reference_text=reference_text,
        model_path=model_path,
        mode=mode,
    )
    cache = load_cache_index(cache_index_path)
    cache_entry = cache["fingerprints"].get(fingerprint)
    if isinstance(cache_entry, dict):
        prompt_cache_path = cache_entry.get("prompt_cache_path")
        if isinstance(prompt_cache_path, str) and prompt_cache_path.strip():
            prompt_path = Path(prompt_cache_path)
            if prompt_path.exists():
                try:
                    prompt_cache = torch.load(prompt_path, map_location="cpu", weights_only=False)
                    if isinstance(prompt_cache, dict):
                        return prompt_cache, fingerprint, True
                except Exception:
                    pass

    if mode == "basic_clone":
        prompt_cache = model.tts_model.build_prompt_cache(
            reference_wav_path=str(reference_audio_path),
            trim_silence_vad=False,
        )
    elif mode == "hifi_clone":
        prompt_cache = model.tts_model.build_prompt_cache(
            prompt_text=reference_text,
            prompt_wav_path=str(reference_audio_path),
            reference_wav_path=str(reference_audio_path),
            trim_silence_vad=False,
        )
    else:
        raise ValueError(f"Unsupported VoxCPM2 mode: {mode}")
    prompt_path = prompts_dir / f"{fingerprint}.pt"
    ensure_out_dir(prompt_path.parent)
    torch.save(prompt_cache, prompt_path)
    cache["fingerprints"][fingerprint] = {
        "prompt_cache_path": str(prompt_path),
        "model_path": str(model_path.resolve()),
        "mode": mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_cache_index(cache, cache_index_path)
    return prompt_cache, fingerprint, False


def synthesize_voxcpm2_tts(
    *,
    text: str,
    reference_audio_path: Path,
    reference_text: str,
    session_dir: Path,
    mode: str,
    cache_index_path: Path,
    prompts_dir: Path,
    model_path: Path = LOCAL_VOXCPM2_MODEL_PATH,
) -> tuple[Path, str, bool]:
    model = load_local_voxcpm2_model(model_path=model_path)
    prompt_cache, fingerprint, used_cached_prompt = create_or_load_voxcpm2_prompt_cache(
        model=model,
        reference_audio_path=reference_audio_path,
        reference_text=reference_text,
        mode=mode,
        cache_index_path=cache_index_path,
        prompts_dir=prompts_dir,
        model_path=model_path,
    )
    wav, _, _ = model.tts_model.generate_with_prompt_cache(
        target_text=text,
        prompt_cache=prompt_cache,
        inference_timesteps=LOCAL_VOXCPM2_INFERENCE_TIMESTEPS,
        cfg_value=LOCAL_VOXCPM2_CFG_VALUE,
        retry_badcase=True,
    )
    if not isinstance(wav, torch.Tensor):
        raise RuntimeError("VoxCPM2 returned an invalid waveform tensor.")
    wav_np = wav.detach().cpu().float().numpy()
    if wav_np.ndim == 2 and wav_np.shape[0] == 1:
        wav_np = wav_np[0]
    output_audio_path = session_dir / "synthesized.wav"
    sf.write(str(output_audio_path), wav_np, model.tts_model.sample_rate, subtype="PCM_16")
    return output_audio_path, fingerprint, used_cached_prompt


def run_pipeline(
    *,
    role: str,
    transcript_text: str,
    voice_description: str,
    backend: str,
    excel_path: Path | None,
    final_ranking_prompt_path: Path,
    reference_window: int,
    gap_ms: int,
    reference_min_seconds: float,
    target_model: str,
    reference_language: str,
    play: bool,
) -> dict[str, Any]:
    if reference_window < 1:
        raise ValueError("reference_window must be at least 1.")
    if gap_ms < 0:
        raise ValueError("gap_ms cannot be negative.")
    if reference_min_seconds <= 0:
        raise ValueError("reference_min_seconds must be greater than 0.")
    if backend not in BACKEND_CHOICES:
        raise ValueError(f"Unsupported backend: {backend}")

    role_paths = ensure_role_excel(role)
    if excel_path is None:
        excel_path = role_paths.excel_path
    excel_path = excel_path.resolve()
    output_paths = make_role_output_paths(role_paths)
    session_dir = make_session_dir(output_paths["tts_runs_dir"])

    retrieval_result = run_double_keyword_retrieval(
        role=role_paths.role,
        transcript_text=transcript_text,
        voice_description=voice_description,
        excel_path=excel_path,
        final_ranking_prompt_path=final_ranking_prompt_path,
        top_k_second=10,
        top_k_final=10,
    )
    write_json(session_dir / "retrieval_result.json", retrieval_result)
    print_retrieval_result(retrieval_result)

    reference_bundle = build_reference_audio(
        retrieval_result["top_results"],
        reference_window=reference_window,
        gap_ms=gap_ms,
        reference_min_seconds=reference_min_seconds,
        session_dir=session_dir,
    )

    api_fingerprint = compute_reference_fingerprint(
        reference_audio_path=reference_bundle["reference_audio_path"],
        reference_text=reference_bundle["reference_text"],
        target_model=target_model,
        language=reference_language,
    )
    voice: str | None = None
    used_cached_voice = False
    retried_voice_creation = False
    voxcpm2_prompt_fingerprint: str | None = None
    used_cached_voxcpm2_prompt = False
    tts_response_payload: dict[str, Any] | None = None

    if backend == "api":
        cache = load_cache_index(output_paths["voice_cache_path"])
        voice, used_cache, enrollment_response = get_or_create_voice(
            fingerprint=api_fingerprint,
            reference_audio_path=reference_bundle["reference_audio_path"],
            reference_text=reference_bundle["reference_text"],
            target_model=target_model,
            language=reference_language,
            cache_path=output_paths["voice_cache_path"],
            cache=cache,
        )
        used_cached_voice = used_cache
        if enrollment_response is not None:
            write_json(session_dir / "voice_enrollment_response.json", enrollment_response)

        while True:
            try:
                audio_url, tts_response = synthesize_api_tts(
                    text=transcript_text,
                    voice=voice,
                    target_model=target_model,
                )
                tts_response_payload = to_jsonable(tts_response)
                break
            except Exception:
                if used_cached_voice and not retried_voice_creation:
                    retried_voice_creation = True
                    remove_cached_voice(api_fingerprint, cache, output_paths["voice_cache_path"])
                    voice, used_cached_voice, enrollment_response = get_or_create_voice(
                        fingerprint=api_fingerprint,
                        reference_audio_path=reference_bundle["reference_audio_path"],
                        reference_text=reference_bundle["reference_text"],
                        target_model=target_model,
                        language=reference_language,
                        cache_path=output_paths["voice_cache_path"],
                        cache=cache,
                    )
                    if enrollment_response is not None:
                        write_json(session_dir / "voice_enrollment_response_retry.json", enrollment_response)
                    continue
                raise

        output_audio_path = choose_output_audio_path(session_dir, audio_url)
        content_type = download_file(audio_url, output_audio_path)
        actual_output_audio_path = choose_output_audio_path(session_dir, audio_url, content_type)
        if actual_output_audio_path != output_audio_path:
            output_audio_path.replace(actual_output_audio_path)
            output_audio_path = actual_output_audio_path
        if tts_response_payload is not None:
            write_json(session_dir / "tts_response.json", tts_response_payload)
    else:
        voxcpm2_mode = "basic_clone" if backend == "voxcpm2_local_basic" else "hifi_clone"
        output_audio_path, voxcpm2_prompt_fingerprint, used_cached_voxcpm2_prompt = synthesize_voxcpm2_tts(
            text=transcript_text,
            reference_audio_path=reference_bundle["reference_audio_path"],
            reference_text=reference_bundle["reference_text"],
            session_dir=session_dir,
            mode=voxcpm2_mode,
            cache_index_path=output_paths["voxcpm2_prompt_cache_index_path"],
            prompts_dir=output_paths["voxcpm2_prompts_dir"],
        )

    target_model_value = target_model if backend == "api" else str(LOCAL_VOXCPM2_MODEL_PATH.resolve())
    run_summary = {
        "role": role_paths.role,
        "role_audio_dir": str(role_paths.audio_dir),
        "role_excel_path": str(excel_path),
        "backend": backend,
        "input_transcript": transcript_text,
        "voice_description": voice_description,
        "session_dir": str(session_dir),
        "reference_audio_path": str(reference_bundle["reference_audio_path"]),
        "reference_text_path": str(reference_bundle["reference_text_path"]),
        "reference_duration_seconds": reference_bundle["duration_seconds"],
        "reference_language": reference_language,
        "target_model": target_model_value,
        "voice": voice,
        "voice_cache_fingerprint": api_fingerprint,
        "used_cached_voice": used_cached_voice if backend == "api" and not retried_voice_creation else False,
        "retried_voice_creation": retried_voice_creation,
        "voxcpm2_prompt_fingerprint": voxcpm2_prompt_fingerprint,
        "used_cached_voxcpm2_prompt": used_cached_voxcpm2_prompt,
        "voxcpm2_mode": (
            "basic_clone"
            if backend == "voxcpm2_local_basic"
            else "hifi_clone"
            if backend == "voxcpm2_local_hifi"
            else None
        ),
        "synthesized_audio_path": str(output_audio_path),
        "selected_reference_results": reference_bundle["selected_results"],
    }
    write_json(session_dir / "run_summary.json", run_summary)

    if play:
        print(f"playing_audio={output_audio_path}")
        play_audio(output_audio_path)

    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="角色化双层检索 + 声音复刻 + TTS 配音")
    parser.add_argument("--role", required=True, help="角色名，例如 莫宁 或 尤诺")
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default=DEFAULT_BACKEND, help="TTS backend mode")
    parser.add_argument("--transcript-text", default=None, help="待合成的配音台词")
    parser.add_argument("--voice-description", default=None, help="用于检索排序的希望配音描述")
    parser.add_argument("--excel", default=None, help="可选，覆盖角色默认 Excel 路径")
    parser.add_argument(
        "--final-ranking-prompt",
        default=str(DEFAULT_FINAL_RANKING_PROMPT_PATH),
        help="最终排序 prompt 模板路径",
    )
    parser.add_argument("--reference-window", type=int, default=DEFAULT_REFERENCE_WINDOW, help="按排序最多尝试前几条参考音频")
    parser.add_argument("--gap-ms", type=int, default=DEFAULT_GAP_MS, help="参考音频之间的静音时长")
    parser.add_argument(
        "--reference-min-seconds",
        type=float,
        default=DEFAULT_REFERENCE_MIN_SECONDS,
        help="参考音频累计时长超过该值时停止，包含插入的静音时长",
    )
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL, help="API 模式下的 TTS 目标模型")
    parser.add_argument("--reference-language", default=DEFAULT_REFERENCE_LANGUAGE, help="API 模式下的声音复刻语言")
    parser.add_argument("--play", dest="play", action="store_true", help="合成后自动播放")
    parser.add_argument("--no-play", dest="play", action="store_false", help="合成后不播放")
    parser.set_defaults(play=True)
    args = parser.parse_args()

    result = run_pipeline(
        role=args.role,
        transcript_text=read_required_text(args.transcript_text, "请输入配音台词: ", "配音台词不能为空。"),
        voice_description=read_required_text(
            args.voice_description,
            "请输入希望的配音描述: ",
            "配音描述不能为空。",
        ),
        backend=args.backend,
        excel_path=Path(args.excel).resolve() if args.excel else None,
        final_ranking_prompt_path=Path(args.final_ranking_prompt).resolve(),
        reference_window=args.reference_window,
        gap_ms=args.gap_ms,
        reference_min_seconds=args.reference_min_seconds,
        target_model=args.target_model,
        reference_language=args.reference_language,
        play=args.play,
    )

    print(f"role={result['role']}")
    print(f"backend={result['backend']}")
    print(f"voice={result['voice']}")
    print(f"role_excel_path={result['role_excel_path']}")
    print(f"reference_audio_path={result['reference_audio_path']}")
    print(f"reference_text_path={result['reference_text_path']}")
    print(f"synthesized_audio_path={result['synthesized_audio_path']}")
    print(f"session_dir={result['session_dir']}")


if __name__ == "__main__":
    main()
