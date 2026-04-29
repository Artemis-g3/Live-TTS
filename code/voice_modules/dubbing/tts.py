from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import dashscope
import numpy as np
import requests
import soundfile as sf
import torch
from dashscope import MultiModalConversation
from voxcpm import VoxCPM

from GUI.config import PROJECT_ROOT, to_jsonable
from voice_modules.dubbing.retrieval import COL_AUDIO_FILE, COL_TRANSCRIPT, get_api_key, run_double_keyword_retrieval
from voice_modules.dubbing.output_overview import tts_retrieval_cache_dir, tts_runs_dir
from voice_modules.role_library.role_library import resolve_role_paths


VOICE_ENROLLMENT_MODEL = "qwen-voice-enrollment"
VOICE_ENROLLMENT_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
DEFAULT_TARGET_MODEL = "qwen3-tts-vc-2026-01-22"
BACKEND_CHOICES = ("api", "voxcpm2_local_basic", "voxcpm2_local_hifi")
DEFAULT_REFERENCE_LANGUAGE = "zh"
DEFAULT_TTS_LANGUAGE_TYPE = "Chinese"
LOCAL_VOXCPM2_INFERENCE_TIMESTEPS = 20
LOCAL_VOXCPM2_CFG_VALUE = 2.0


_LOCAL_MODEL: VoxCPM | None = None


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def run_file_path(session_dir: Path, stem: str, suffix: str) -> Path:
    return session_dir / f"{stem}_{session_dir.name}{suffix}"


def usage_to_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def make_session_dir(base_dir: Path) -> Path:
    ensure_out_dir(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base_dir / stamp
    suffix = 1
    while path.exists():
        path = base_dir / f"{stamp}_{suffix:02d}"
        suffix += 1
    ensure_out_dir(path)
    return path


def unique_session_path(session_dir: Path, stem: str, suffix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = session_dir / f"{stem}_{stamp}{suffix}"
    index = 1
    while path.exists():
        path = session_dir / f"{stem}_{stamp}_{index:02d}{suffix}"
        index += 1
    return path


def build_reference_audio(
    top_results: list[dict[str, Any]],
    *,
    reference_window: int,
    gap_ms: int,
    reference_min_seconds: float,
    session_dir: Path,
    selected_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not top_results:
        raise RuntimeError("没有可用于声音复刻的检索结果。")
    segments: list[np.ndarray] = []
    selected: list[dict[str, Any]] = []
    texts: list[str] = []
    sample_rate: int | None = None
    total_samples = 0

    source_rows = selected_results if selected_results is not None else top_results[:reference_window]
    for row in source_rows:
        audio_path = Path(str(row["audio_path"]))
        if not audio_path.exists():
            raise FileNotFoundError(f"参考音频不存在: {audio_path}")
        audio_data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        duration = len(audio_data) / float(sr)
        if selected_results is None and duration > reference_min_seconds:
            continue
        if sample_rate is None:
            sample_rate = int(sr)
        elif int(sr) != sample_rate:
            raise RuntimeError(f"参考音频采样率不一致: {audio_path.name}={sr}, expected={sample_rate}")
        if segments:
            gap = np.zeros(int(round(sample_rate * gap_ms / 1000.0)), dtype=np.float32)
            segments.append(gap)
            total_samples += len(gap)
        segments.append(audio_data.astype(np.float32))
        total_samples += len(audio_data)
        selected.append(row)
        texts.append(str(row[COL_TRANSCRIPT]))
        if selected_results is None and total_samples / sample_rate > reference_min_seconds:
            break

    if sample_rate is None or not segments:
        raise RuntimeError("未能生成参考音频。")
    wav = np.concatenate(segments).astype(np.float32)
    reference_audio_path = run_file_path(session_dir, "reference_audio", ".wav")
    sf.write(str(reference_audio_path), wav, sample_rate, subtype="PCM_16")
    reference_text = "\n".join(texts)
    reference_text_path = run_file_path(session_dir, "reference_text", ".txt")
    reference_text_path.write_text(reference_text, encoding="utf-8")
    return {
        "selected_results": selected,
        "reference_audio_path": reference_audio_path,
        "reference_text_path": reference_text_path,
        "reference_text": reference_text,
        "sample_rate": sample_rate,
        "duration_seconds": len(wav) / sample_rate,
    }


def make_data_url(path: Path, mime_type: str = "audio/wav") -> str:
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def compute_reference_fingerprint(reference_audio_path: Path, reference_text: str, target_model: str, language: str) -> str:
    payload = hashlib.sha256()
    payload.update(reference_audio_path.read_bytes())
    payload.update(reference_text.encode("utf-8"))
    payload.update(target_model.encode("utf-8"))
    payload.update(language.encode("utf-8"))
    return payload.hexdigest()


def load_cache_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"fingerprints": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"fingerprints": {}}
    fingerprints = data.get("fingerprints", {})
    return {"fingerprints": fingerprints if isinstance(fingerprints, dict) else {}}


def save_cache_index(path: Path, cache: dict[str, Any]) -> None:
    ensure_out_dir(path.parent)
    write_json(path, cache)


def create_voice(reference_audio_path: Path, reference_text: str, target_model: str, language: str, preferred_name: str, voice_enrollment_model: str = VOICE_ENROLLMENT_MODEL) -> dict[str, Any]:
    response = requests.post(
        VOICE_ENROLLMENT_ENDPOINT,
        headers={"Authorization": f"Bearer {get_api_key()}", "Content-Type": "application/json"},
        json={
            "model": voice_enrollment_model,
            "input": {
                "action": "create",
                "target_model": target_model,
                "audio": {"data": make_data_url(reference_audio_path)},
                "text": reference_text,
                "language": language,
                "preferred_name": preferred_name,
            },
        },
        timeout=300,
    )
    if not response.ok:
        raise RuntimeError(f"声音复刻请求失败: status={response.status_code} body={response.text}")
    data = response.json()
    if data.get("code"):
        raise RuntimeError(f"{data['code']}: {data.get('message', '')}")
    voice = (data.get("output", {}) or {}).get("voice")
    if isinstance(voice, dict):
        voice = voice.get("name") or voice.get("voice") or voice.get("id")
    if not isinstance(voice, str) or not voice.strip():
        raise RuntimeError(f"声音复刻返回中缺少 voice 字段: {data}")
    return {"voice": voice.strip(), "response": data}


def get_or_create_voice(fingerprint: str, reference_audio_path: Path, reference_text: str, target_model: str, language: str, cache_path: Path, voice_enrollment_model: str = VOICE_ENROLLMENT_MODEL) -> tuple[str, bool, dict[str, Any] | None]:
    cache = load_cache_index(cache_path)
    entry = cache["fingerprints"].get(fingerprint)
    if isinstance(entry, dict) and isinstance(entry.get("voice"), str) and entry["voice"].strip():
        return entry["voice"].strip(), True, None
    preferred_name = f"vc_{fingerprint[:10]}"
    created = create_voice(reference_audio_path, reference_text, target_model, language, preferred_name, voice_enrollment_model)
    cache["fingerprints"][fingerprint] = {
        "voice": created["voice"],
        "preferred_name": preferred_name,
        "target_model": target_model,
        "voice_enrollment_model": voice_enrollment_model,
        "language": language,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_cache_index(cache_path, cache)
    return created["voice"], False, created["response"]


def synthesize_api_tts(text: str, voice: str, target_model: str) -> tuple[str, Any]:
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
    audio = getattr(getattr(response, "output", None), "audio", None)
    audio_url = getattr(audio, "url", None) if audio is not None else None
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
        return run_file_path(session_dir, "synthesized", ".mp3")
    if "wav" in content_type or "wave" in content_type:
        return run_file_path(session_dir, "synthesized", ".wav")
    suffix = Path(urlparse(audio_url).path).suffix.lower()
    return run_file_path(session_dir, "synthesized", ".mp3" if suffix == ".mp3" else ".wav")


def load_local_model(model_path: Path) -> VoxCPM:
    global _LOCAL_MODEL
    if _LOCAL_MODEL is not None:
        return _LOCAL_MODEL
    if not model_path.exists():
        raise FileNotFoundError(f"VoxCPM2 模型目录不存在: {model_path}")
    _LOCAL_MODEL = VoxCPM.from_pretrained(str(model_path), load_denoiser=False, local_files_only=True, optimize=True)
    return _LOCAL_MODEL


def compute_local_prompt_fingerprint(reference_audio_path: Path, reference_text: str, model_path: Path, mode: str) -> str:
    payload = hashlib.sha256()
    payload.update(reference_audio_path.read_bytes())
    payload.update(reference_text.encode("utf-8"))
    payload.update(str(model_path.resolve()).encode("utf-8"))
    payload.update(mode.encode("utf-8"))
    return payload.hexdigest()


def create_or_load_prompt_cache(model: VoxCPM, reference_audio_path: Path, reference_text: str, mode: str, cache_index_path: Path, prompts_dir: Path, model_path: Path) -> tuple[dict[str, Any], str, bool]:
    fingerprint = compute_local_prompt_fingerprint(reference_audio_path, reference_text, model_path, mode)
    cache = load_cache_index(cache_index_path)
    entry = cache["fingerprints"].get(fingerprint)
    if isinstance(entry, dict):
        prompt_cache_path = entry.get("prompt_cache_path")
        if isinstance(prompt_cache_path, str) and Path(prompt_cache_path).exists():
            try:
                prompt_cache = torch.load(prompt_cache_path, map_location="cpu", weights_only=False)
                if isinstance(prompt_cache, dict):
                    return prompt_cache, fingerprint, True
            except Exception:
                pass
    if mode == "basic_clone":
        prompt_cache = model.tts_model.build_prompt_cache(reference_wav_path=str(reference_audio_path), trim_silence_vad=False)
    elif mode == "hifi_clone":
        prompt_cache = model.tts_model.build_prompt_cache(
            prompt_text=reference_text,
            prompt_wav_path=str(reference_audio_path),
            reference_wav_path=str(reference_audio_path),
            trim_silence_vad=False,
        )
    else:
        raise ValueError(f"不支持的 VoxCPM2 模式: {mode}")
    ensure_out_dir(prompts_dir)
    prompt_path = prompts_dir / f"{fingerprint}.pt"
    torch.save(prompt_cache, prompt_path)
    cache["fingerprints"][fingerprint] = {
        "prompt_cache_path": str(prompt_path),
        "model_path": str(model_path.resolve()),
        "mode": mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_cache_index(cache_index_path, cache)
    return prompt_cache, fingerprint, False


def compose_voxcpm2_text(transcript_text: str, synthesis_voice_guidance: str) -> str:
    guidance = synthesis_voice_guidance.strip()
    text = transcript_text.strip()
    if not guidance:
        return text
    if guidance.startswith("(") and guidance.endswith(")"):
        prefix = guidance
    else:
        prefix = f"({guidance})"
    return f"{prefix}{text}"


def synthesize_voxcpm2_tts(
    text: str,
    reference_audio_path: Path,
    reference_text: str,
    session_dir: Path,
    mode: str,
    cache_index_path: Path,
    prompts_dir: Path,
    model_path: Path,
    diffusion_steps: int = LOCAL_VOXCPM2_INFERENCE_TIMESTEPS,
    cfg_value: float = LOCAL_VOXCPM2_CFG_VALUE,
) -> tuple[Path, str, bool]:
    model = load_local_model(model_path)
    prompt_cache, fingerprint, used_cache = create_or_load_prompt_cache(model, reference_audio_path, reference_text, mode, cache_index_path, prompts_dir, model_path)
    wav, _, _ = model.tts_model.generate_with_prompt_cache(
        target_text=text,
        prompt_cache=prompt_cache,
        inference_timesteps=diffusion_steps,
        cfg_value=cfg_value,
        retry_badcase=True,
    )
    if not isinstance(wav, torch.Tensor):
        raise RuntimeError("VoxCPM2 返回了无效音频张量。")
    wav_np = wav.detach().cpu().float().numpy()
    if wav_np.ndim == 2 and wav_np.shape[0] == 1:
        wav_np = wav_np[0]
    output_path = run_file_path(session_dir, "synthesized", ".wav")
    sf.write(str(output_path), wav_np, model.tts_model.sample_rate, subtype="PCM_16")
    return output_path, fingerprint, used_cache


def play_audio(path: Path) -> None:
    candidates = [Path(r"C:\Program Files\ffmpeg\bin\ffplay.exe"), Path("ffplay.exe")]
    for candidate in candidates:
        try:
            subprocess.run([str(candidate), "-autoexit", "-nodisp", "-loglevel", "error", str(path)], check=True)
            return
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError:
            continue
    if path.suffix.lower() == ".wav":
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return
    os.startfile(str(path))


def run_retrieval_stage(
    *,
    role: str,
    transcript_text: str,
    retrieval_voice_guidance: str,
    rerank_model: str = "qwen3-rerank",
    text_model: str = "qwen3.6-flash",
) -> dict[str, Any]:
    if not transcript_text.strip():
        raise ValueError("配音台词不能为空。")
    if not retrieval_voice_guidance.strip():
        raise ValueError("检索声音指导不能为空。")

    role_paths = resolve_role_paths(role)
    session_dir = make_session_dir(tts_retrieval_cache_dir(role_paths.role))
    print(f"session_dir={session_dir}", flush=True)
    print("[Retrieval] 开始检索参考音频。", flush=True)

    retrieval_result = run_double_keyword_retrieval(
        role=role_paths.role,
        transcript_text=transcript_text,
        voice_description=retrieval_voice_guidance,
        top_k_second=10,
        top_k_final=10,
        rerank_model=rerank_model,
        text_model=text_model,
    )
    retrieval_result_path = session_dir / "retrieval_result.json"
    write_json(retrieval_result_path, retrieval_result)
    print(f"retrieval_result_path={retrieval_result_path}", flush=True)
    token_usage = retrieval_result.get("model_token_usage", {})
    rerank_usage = token_usage.get("rerank", {})
    if int(rerank_usage.get("total_tokens", 0) or 0) > 0:
        print(
            f"[Retrieval] API token: model={rerank_usage.get('model', rerank_model)} "
            f"input={rerank_usage.get('input_tokens', 0)} total={rerank_usage.get('total_tokens', 0)}",
            flush=True,
        )
    second_layer_usage = token_usage.get("second_layer", {})
    if int(second_layer_usage.get("total_tokens", 0) or 0) > 0:
        print(
            f"[Retrieval] API token: model={second_layer_usage.get('model', text_model)} "
            f"prompt={second_layer_usage.get('prompt_tokens', 0)} "
            f"completion={second_layer_usage.get('completion_tokens', 0)} "
            f"total={second_layer_usage.get('total_tokens', 0)}",
            flush=True,
        )
    for row in retrieval_result.get("top_results", []):
        duration_seconds = float(row.get("duration_seconds", 0) or 0)
        print(
            f"[Retrieval] rank={row.get('rank')} audio={row.get(COL_AUDIO_FILE, '')} "
            f"duration={duration_seconds:.1f}s path={row.get('audio_path', '')}",
            flush=True,
        )
    print("[Retrieval] 检索完成。", flush=True)
    return {
        "role": role_paths.role,
        "session_dir": str(session_dir),
        "retrieval_result_path": str(retrieval_result_path),
        "retrieval_voice_guidance": retrieval_voice_guidance,
        "top_results": retrieval_result["top_results"],
    }


def run_synthesis_from_retrieval(
    *,
    role: str,
    transcript_text: str,
    retrieval_voice_guidance: str,
    synthesis_voice_guidance: str,
    backend: str,
    reference_window: int,
    reference_min_seconds: float,
    gap_ms: int,
    voxcpm2_diffusion_steps: int,
    voxcpm2_cfg_value: float,
    voxcpm2_model_path: Path,
    retrieval_result_path: Path,
    session_dir: Path,
    selected_results: list[dict[str, Any]] | None = None,
    target_model: str = DEFAULT_TARGET_MODEL,
    voice_enrollment_model: str = VOICE_ENROLLMENT_MODEL,
    reference_language: str = DEFAULT_REFERENCE_LANGUAGE,
    play: bool = False,
) -> dict[str, Any]:
    if backend not in BACKEND_CHOICES:
        raise ValueError(f"不支持的后端: {backend}")
    if not transcript_text.strip():
        raise ValueError("配音台词不能为空。")
    if not retrieval_voice_guidance.strip():
        raise ValueError("检索声音指导不能为空。")
    if not retrieval_result_path.exists():
        raise FileNotFoundError(f"检索结果文件不存在: {retrieval_result_path}")
    if voxcpm2_diffusion_steps < 1:
        raise ValueError("VoxCPM2 扩散轮数必须大于 0。")
    if not 1.0 <= voxcpm2_cfg_value <= 3.0:
        raise ValueError("VoxCPM2 cfg_value 必须在 1.0 到 3.0 之间。")

    role_paths = resolve_role_paths(role)
    ensure_out_dir(session_dir)
    retrieval_result = json.loads(retrieval_result_path.read_text(encoding="utf-8"))
    run_dir = make_session_dir(tts_runs_dir(role_paths.role))
    ensure_out_dir(run_dir)
    run_retrieval_result_path = run_dir / "retrieval_result.json"
    shutil.copy2(retrieval_result_path, run_retrieval_result_path)
    print("[TTS] 开始根据检索结果合成配音。", flush=True)

    reference_bundle = build_reference_audio(
        retrieval_result["top_results"],
        reference_window=reference_window,
        gap_ms=gap_ms,
        reference_min_seconds=reference_min_seconds,
        session_dir=run_dir,
        selected_results=selected_results,
    )
    print(f"reference_audio_path={reference_bundle['reference_audio_path']}", flush=True)

    voice = None
    used_cached_voice = False
    voxcpm2_prompt_fingerprint = None
    used_cached_voxcpm2_prompt = False
    tts_response_payload = None

    if backend == "api":
        cache_path = role_paths.output_dir / "voice_cache.json"
        fingerprint = compute_reference_fingerprint(
            reference_bundle["reference_audio_path"],
            reference_bundle["reference_text"],
            target_model,
            reference_language,
        )
        voice, used_cached_voice, enrollment_response = get_or_create_voice(
            fingerprint,
            reference_bundle["reference_audio_path"],
            reference_bundle["reference_text"],
            target_model,
            reference_language,
            cache_path,
            voice_enrollment_model,
        )
        if enrollment_response is not None:
            write_json(run_file_path(run_dir, "voice_enrollment_response", ".json"), enrollment_response)
            enrollment_usage = enrollment_response.get("usage", {}) if isinstance(enrollment_response, dict) else {}
            if int(enrollment_usage.get("total_tokens", 0) or 0) > 0:
                print(
                    f"[TTS] API token: model={voice_enrollment_model} "
                    f"prompt={enrollment_usage.get('prompt_tokens', 0)} "
                    f"completion={enrollment_usage.get('completion_tokens', 0)} "
                    f"total={enrollment_usage.get('total_tokens', 0)}",
                    flush=True,
                )
        audio_url, tts_response = synthesize_api_tts(transcript_text, voice, target_model)
        tts_response_payload = to_jsonable(tts_response)
        tts_usage = usage_to_dict(getattr(tts_response, "usage", None))
        if tts_usage["total_tokens"] > 0:
            print(
                f"[TTS] API token: model={target_model} prompt={tts_usage['prompt_tokens']} "
                f"completion={tts_usage['completion_tokens']} total={tts_usage['total_tokens']}",
                flush=True,
            )
        output_audio_path = choose_output_audio_path(run_dir, audio_url)
        content_type = download_file(audio_url, output_audio_path)
        final_suffix = ".mp3" if "mpeg" in content_type.lower() or "mp3" in content_type.lower() else ".wav"
        final_path = output_audio_path.with_suffix(final_suffix)
        if final_path != output_audio_path:
            output_audio_path.replace(final_path)
            output_audio_path = final_path
        write_json(run_file_path(run_dir, "tts_response", ".json"), tts_response_payload)
        api_fingerprint = fingerprint
    else:
        mode = "basic_clone" if backend == "voxcpm2_local_basic" else "hifi_clone"
        basic_guidance = synthesis_voice_guidance if backend == "voxcpm2_local_basic" else ""
        voxcpm2_text = compose_voxcpm2_text(transcript_text, basic_guidance)
        print(f"[TTS] VoxCPM2 cfg_value={voxcpm2_cfg_value:.2f}", flush=True)
        if basic_guidance.strip():
            print(f"[TTS] VoxCPM2 synthesis_guidance={basic_guidance.strip()}", flush=True)
        output_audio_path, voxcpm2_prompt_fingerprint, used_cached_voxcpm2_prompt = synthesize_voxcpm2_tts(
            text=voxcpm2_text,
            reference_audio_path=reference_bundle["reference_audio_path"],
            reference_text=reference_bundle["reference_text"],
            session_dir=run_dir,
            mode=mode,
            cache_index_path=role_paths.output_dir / "voxcpm2_prompt_cache.json",
            prompts_dir=role_paths.output_dir / "voxcpm2_prompt_cache",
            model_path=voxcpm2_model_path,
            diffusion_steps=voxcpm2_diffusion_steps,
            cfg_value=voxcpm2_cfg_value,
        )
        api_fingerprint = ""

    run_summary = {
        "role": role_paths.role,
        "backend": backend,
        "tts_target_model": target_model,
        "voice_enrollment_model": voice_enrollment_model,
        "input_transcript": transcript_text,
        "voice_description": retrieval_voice_guidance,
        "retrieval_voice_guidance": retrieval_voice_guidance,
        "synthesis_voice_guidance": synthesis_voice_guidance,
        "session_dir": str(run_dir),
        "retrieval_result_path": str(run_retrieval_result_path),
        "retrieval_source_session_dir": str(session_dir),
        "retrieval_source_result_path": str(retrieval_result_path),
        "reference_audio_path": str(reference_bundle["reference_audio_path"]),
        "reference_text_path": str(reference_bundle["reference_text_path"]),
        "reference_duration_seconds": reference_bundle["duration_seconds"],
        "voxcpm2_diffusion_steps": voxcpm2_diffusion_steps,
        "voxcpm2_cfg_value": voxcpm2_cfg_value,
        "synthesized_audio_path": str(output_audio_path),
        "voice": voice,
        "voice_cache_fingerprint": api_fingerprint,
        "used_cached_voice": used_cached_voice,
        "voxcpm2_prompt_fingerprint": voxcpm2_prompt_fingerprint,
        "used_cached_voxcpm2_prompt": used_cached_voxcpm2_prompt,
        "selected_reference_results": reference_bundle["selected_results"],
    }
    run_summary_path = run_dir / "run_summary.json"
    run_summary["run_summary_path"] = str(run_summary_path)
    write_json(run_summary_path, run_summary)
    print(f"synthesized_audio_path={output_audio_path}", flush=True)
    print("[TTS] 合成完成。", flush=True)
    if play:
        play_audio(output_audio_path)
    return run_summary


def run_tts_pipeline(
    *,
    role: str,
    transcript_text: str,
    retrieval_voice_guidance: str,
    synthesis_voice_guidance: str,
    backend: str,
    reference_window: int,
    reference_min_seconds: float,
    gap_ms: int,
    voxcpm2_diffusion_steps: int = LOCAL_VOXCPM2_INFERENCE_TIMESTEPS,
    voxcpm2_cfg_value: float = LOCAL_VOXCPM2_CFG_VALUE,
    voxcpm2_model_path: Path,
    selected_results: list[dict[str, Any]] | None = None,
    target_model: str = DEFAULT_TARGET_MODEL,
    voice_enrollment_model: str = VOICE_ENROLLMENT_MODEL,
    rerank_model: str = "qwen3-rerank",
    text_model: str = "qwen3.6-flash",
    reference_language: str = DEFAULT_REFERENCE_LANGUAGE,
    play: bool = False,
) -> dict[str, Any]:
    if backend not in BACKEND_CHOICES:
        raise ValueError(f"不支持的后端: {backend}")
    if not transcript_text.strip():
        raise ValueError("配音台词不能为空。")
    if not retrieval_voice_guidance.strip():
        raise ValueError("检索声音指导不能为空。")

    retrieval_summary = run_retrieval_stage(
        role=role,
        transcript_text=transcript_text,
        retrieval_voice_guidance=retrieval_voice_guidance,
        rerank_model=rerank_model,
        text_model=text_model,
    )
    return run_synthesis_from_retrieval(
        role=role,
        transcript_text=transcript_text,
        retrieval_voice_guidance=retrieval_voice_guidance,
        synthesis_voice_guidance=synthesis_voice_guidance,
        backend=backend,
        reference_window=reference_window,
        reference_min_seconds=reference_min_seconds,
        gap_ms=gap_ms,
        voxcpm2_diffusion_steps=voxcpm2_diffusion_steps,
        voxcpm2_cfg_value=voxcpm2_cfg_value,
        voxcpm2_model_path=voxcpm2_model_path,
        selected_results=selected_results,
        target_model=target_model,
        voice_enrollment_model=voice_enrollment_model,
        reference_language=reference_language,
        play=play,
        retrieval_result_path=Path(retrieval_summary["retrieval_result_path"]),
        session_dir=Path(retrieval_summary["session_dir"]),
    )
