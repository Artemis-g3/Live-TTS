from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from voice_modules.role_library.role_library import resolve_role_paths


REQUIRED_FIXED_FILENAMES = {"run_summary.json", "retrieval_result.json"}
OPTIONAL_PREFIXES = ("tts_response", "voice_enrollment_response")
REFERENCE_AUDIO_PREFIX = "reference_audio"
REFERENCE_TEXT_PREFIX = "reference_text"
SYNTHESIZED_PREFIX = "synthesized"


def tts_runs_dir(role: str) -> Path:
    return (resolve_role_paths(role).output_dir / "tts_runs").resolve()


def tts_retrieval_cache_dir(role: str) -> Path:
    return (resolve_role_paths(role).output_dir / "tts_retrieval_cache").resolve()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _run_dirs(role: str) -> list[Path]:
    root = tts_runs_dir(role)
    if not root.exists():
        return []
    return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)


def _is_conforming_run_dir(run_dir: Path) -> bool:
    files = [path for path in run_dir.iterdir() if path.is_file()]
    file_names = {path.name for path in files}
    if not REQUIRED_FIXED_FILENAMES.issubset(file_names):
        return False
    synthesized_files = _match_prefixed_files(run_dir, SYNTHESIZED_PREFIX, {".wav", ".mp3"})
    reference_audio_files = _match_prefixed_files(run_dir, REFERENCE_AUDIO_PREFIX, {".wav"})
    reference_text_files = _match_prefixed_files(run_dir, REFERENCE_TEXT_PREFIX, {".txt"})
    if len(synthesized_files) != 1 or len(reference_audio_files) != 1 or len(reference_text_files) != 1:
        return False
    for path in files:
        if path.name in REQUIRED_FIXED_FILENAMES:
            continue
        if path in synthesized_files or path in reference_audio_files or path in reference_text_files:
            continue
        if any(path.stem.startswith(prefix) and path.suffix.lower() == ".json" for prefix in OPTIONAL_PREFIXES):
            continue
        return False
    return True


def _match_prefixed_files(run_dir: Path, prefix: str, suffixes: set[str]) -> list[Path]:
    return sorted(
        [
            path
            for path in run_dir.iterdir()
            if path.is_file()
            and path.stem.startswith(prefix)
            and path.suffix.lower() in suffixes
        ],
        key=lambda path: path.name,
    )


def _single_prefixed_file(run_dir: Path, prefix: str, suffixes: set[str]) -> Path:
    matches = _match_prefixed_files(run_dir, prefix, suffixes)
    if len(matches) != 1:
        raise FileNotFoundError(f"{run_dir} 中未找到唯一的 {prefix} 文件。")
    return matches[0]


def _canonical_run_file(run_dir: Path, prefix: str, suffix: str) -> Path:
    return run_dir / f"{prefix}_{run_dir.name}{suffix}"


def _conforming_run_dirs(role: str) -> list[Path]:
    return [run_dir for run_dir in _run_dirs(role) if _is_conforming_run_dir(run_dir)]


def cleanup_nonconforming_tts_runs(role: str) -> dict[str, Any]:
    deleted_dirs: list[str] = []
    root = tts_runs_dir(role)
    if not root.exists():
        return {"deleted_dirs": deleted_dirs, "deleted_count": 0}
    for run_dir in _run_dirs(role):
        if _is_conforming_run_dir(run_dir):
            continue
        shutil.rmtree(run_dir, ignore_errors=True)
        if not run_dir.exists():
            deleted_dirs.append(str(run_dir.resolve()))
    return {"deleted_dirs": deleted_dirs, "deleted_count": len(deleted_dirs)}


def normalize_tts_run_filenames(role: str) -> dict[str, Any]:
    renamed_files: list[dict[str, str]] = []
    updated_runs = 0
    for run_dir in _conforming_run_dirs(role):
        summary_path = run_dir / "run_summary.json"
        payload = _load_json(summary_path)
        if payload is None:
            continue
        changes_made = False

        synthesized_path = _single_prefixed_file(run_dir, SYNTHESIZED_PREFIX, {".wav", ".mp3"})
        canonical_synthesized_path = _canonical_run_file(run_dir, SYNTHESIZED_PREFIX, synthesized_path.suffix.lower())
        if synthesized_path != canonical_synthesized_path:
            synthesized_path.rename(canonical_synthesized_path)
            renamed_files.append({"from": str(synthesized_path), "to": str(canonical_synthesized_path)})
            synthesized_path = canonical_synthesized_path
            changes_made = True

        reference_audio_path = _single_prefixed_file(run_dir, REFERENCE_AUDIO_PREFIX, {".wav"})
        canonical_reference_audio_path = _canonical_run_file(run_dir, REFERENCE_AUDIO_PREFIX, ".wav")
        if reference_audio_path != canonical_reference_audio_path:
            reference_audio_path.rename(canonical_reference_audio_path)
            renamed_files.append({"from": str(reference_audio_path), "to": str(canonical_reference_audio_path)})
            reference_audio_path = canonical_reference_audio_path
            changes_made = True

        reference_text_path = _single_prefixed_file(run_dir, REFERENCE_TEXT_PREFIX, {".txt"})
        canonical_reference_text_path = _canonical_run_file(run_dir, REFERENCE_TEXT_PREFIX, ".txt")
        if reference_text_path != canonical_reference_text_path:
            reference_text_path.rename(canonical_reference_text_path)
            renamed_files.append({"from": str(reference_text_path), "to": str(canonical_reference_text_path)})
            reference_text_path = canonical_reference_text_path
            changes_made = True

        for prefix in OPTIONAL_PREFIXES:
            optional_matches = _match_prefixed_files(run_dir, prefix, {".json"})
            if len(optional_matches) != 1:
                continue
            optional_path = optional_matches[0]
            canonical_optional_path = _canonical_run_file(run_dir, prefix, ".json")
            if optional_path != canonical_optional_path:
                optional_path.rename(canonical_optional_path)
                renamed_files.append({"from": str(optional_path), "to": str(canonical_optional_path)})
                changes_made = True

        desired_summary_values = {
            "session_dir": str(run_dir.resolve()),
            "retrieval_result_path": str((run_dir / "retrieval_result.json").resolve()),
            "reference_audio_path": str(reference_audio_path.resolve()),
            "reference_text_path": str(reference_text_path.resolve()),
            "synthesized_audio_path": str(synthesized_path.resolve()),
            "run_summary_path": str(summary_path.resolve()),
        }
        for key, expected in desired_summary_values.items():
            if str(payload.get(key, "") or "") != expected:
                payload[key] = expected
                changes_made = True
        if changes_made:
            summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            updated_runs += 1

    return {"updated_runs": updated_runs, "renamed_files": renamed_files}


def list_tts_runs(role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in _conforming_run_dirs(role):
        summary_path = run_dir / "run_summary.json"
        payload = _load_json(summary_path)
        if payload is None:
            continue
        audio_path = _single_prefixed_file(run_dir, SYNTHESIZED_PREFIX, {".wav", ".mp3"})
        reference_audio_path = _single_prefixed_file(run_dir, REFERENCE_AUDIO_PREFIX, {".wav"})
        reference_text_path = _single_prefixed_file(run_dir, REFERENCE_TEXT_PREFIX, {".txt"})
        backend = str(payload.get("backend", "") or "").strip()
        synthesis_voice_guidance = str(payload.get("synthesis_voice_guidance", "") or "").strip()
        if backend != "voxcpm2_local_basic":
            synthesis_voice_guidance = ""
        mtime_source = audio_path if audio_path.exists() else summary_path
        rows.append(
            {
                "语音名称": audio_path.name,
                "配音台词": str(payload.get("input_transcript", "") or "").strip(),
                "检索声音指导文本": str(
                    payload.get("retrieval_voice_guidance", "") or payload.get("voice_description", "") or ""
                ).strip(),
                "合成声音指导文本": synthesis_voice_guidance,
                "audio_path": str(audio_path.resolve()),
                "run_summary_path": str(summary_path.resolve()),
                "session_dir": str(run_dir.resolve()),
                "retrieval_result_path": str((run_dir / "retrieval_result.json").resolve()),
                "reference_audio_path": str(reference_audio_path.resolve()),
                "reference_text_path": str(reference_text_path.resolve()),
                "backend": backend,
                "mtime": mtime_source.stat().st_mtime,
            }
        )
    rows.sort(key=lambda row: (float(row.get("mtime", 0) or 0), str(row.get("语音名称", ""))), reverse=True)
    return rows


def delete_tts_run(role: str, run_summary_path: str) -> dict[str, Any]:
    root = tts_runs_dir(role)
    summary_path = Path(run_summary_path).resolve()
    try:
        summary_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"run_summary 不在角色输出目录内: {summary_path}") from exc
    run_dir = summary_path.parent.resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"run_summary 不存在: {summary_path}")
    deleted_audio_path = ""
    synthesized_files = _match_prefixed_files(run_dir, SYNTHESIZED_PREFIX, {".wav", ".mp3"})
    if synthesized_files:
        deleted_audio_path = str(synthesized_files[0].resolve())
    delete_errors: list[str] = []

    def _onerror(_func, path, exc_info) -> None:
        message = str(exc_info[1]) if len(exc_info) > 1 else "unknown error"
        delete_errors.append(f"{path}: {message}")

    shutil.rmtree(run_dir, onerror=_onerror)
    if run_dir.exists():
        detail = "; ".join(delete_errors) if delete_errors else "unknown error"
        raise RuntimeError(f"删除合成目录失败: {run_dir}; details: {detail}")
    return {
        "deleted_paths": [str(run_dir)],
        "deleted_audio_path": deleted_audio_path,
        "deleted_run_summary_path": str(summary_path),
        "deleted_session_dir": str(run_dir),
        "session_cleared": True,
    }


def latest_tts_run_info(role: str) -> dict[str, str]:
    run_dirs = _conforming_run_dirs(role)
    if not run_dirs:
        return {"latest_tts_run": "", "latest_synthesized_audio": "", "tts_run_count": "0"}
    latest_dir = max(run_dirs, key=lambda path: path.name)
    audio_path = _single_prefixed_file(latest_dir, SYNTHESIZED_PREFIX, {".wav", ".mp3"})
    return {
        "latest_tts_run": latest_dir.name,
        "latest_synthesized_audio": str(audio_path.resolve()) if audio_path.exists() else "",
        "tts_run_count": str(len(run_dirs)),
    }
