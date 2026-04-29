from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from modelscope.pipelines import pipeline

from voice_modules.common.audio_manifest import PROJECT_ROOT, now_tag, read_manifest, resolve_audio_path_for_row, rows_as_dicts, write_manifest


MODEL_DIR = PROJECT_ROOT / "code" / "voice_modules" / "audio_filter" / "speech_campplus_sv_zh-cn_16k-common"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def read_audio(path: Path, sample_rate: int = 16000) -> np.ndarray:
    data, source_sr = sf.read(str(path), dtype="float32")
    if data.ndim == 2:
        data = data[:, 0]
    waveform = torch.from_numpy(data.astype("float32"))
    if source_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sr, sample_rate)
    return waveform.numpy().astype("float32")


def extract_embedding(sv_pipeline: Any, path: Path) -> np.ndarray:
    result = sv_pipeline([read_audio(path)], output_emb=True)
    emb = np.asarray(result["embs"][0], dtype="float32")
    return normalize_vector(emb)


def row_audio_path(row: Any) -> Path:
    return resolve_audio_path_for_row(
        role=row.role,
        audio_path=row.audio_path,
        stored_name=row.stored_name,
        file_name=row.file_name,
    )


def run_speaker_filter(role: str, confirm_threshold: float, review_threshold: float, *, save: bool = False) -> dict[str, Any]:
    if review_threshold >= confirm_threshold:
        raise ValueError("review 阈值必须小于 confirm 阈值。")
    rows = read_manifest(role)
    references = [row for row in rows if row.selected_reference and row_audio_path(row).exists()]
    candidates = [row for row in rows if row.filter_status not in {"deleted", "excluded"} and row_audio_path(row).exists()]
    if len(references) < 2:
        raise RuntimeError("至少需要选择 2 条参考音频。")
    sv_pipeline = pipeline(task="speaker-verification", model=str(MODEL_DIR))
    ref_embeddings = [extract_embedding(sv_pipeline, row_audio_path(row)) for row in references]
    centroid = normalize_vector(np.mean(np.stack(ref_embeddings), axis=0))
    counts = {"confirmed": 0, "review": 0, "reject": 0}
    total = len(candidates)
    print(f"[Filter] 开始运行，共 {total} 条候选音频。", flush=True)
    for index, row in enumerate(candidates, start=1):
        emb = extract_embedding(sv_pipeline, row_audio_path(row))
        ref_scores = [cosine_similarity(emb, ref_emb) for ref_emb in ref_embeddings]
        centroid_score = cosine_similarity(emb, centroid)
        max_ref = max(ref_scores)
        median_ref = float(np.median(ref_scores))
        strong = min(centroid_score, max_ref)
        soft = min(max_ref, max(centroid_score, median_ref))
        if strong >= confirm_threshold:
            status = "confirmed"
        elif soft >= review_threshold:
            status = "review"
        else:
            status = "reject"
        row.filter_status = status
        row.centroid_score = f"{centroid_score:.6f}"
        row.max_ref_score = f"{max_ref:.6f}"
        row.median_ref_score = f"{median_ref:.6f}"
        counts[status] += 1
        if index % 5 == 0 or index == total:
            print(f"[Filter] 已处理 {index}/{total} 条。", flush=True)
    if save:
        write_manifest(role, rows)
    print(
        f"[Filter] 运行完成。confirmed={counts['confirmed']}, review={counts['review']}, reject={counts['reject']}",
        flush=True,
    )
    if not save:
        print("[Filter] 筛选结果尚未写入数据库文件，请在 GUI 中点击“保存结果”。", flush=True)
    return {"role": role, "counts": counts, "rows": rows_as_dicts(rows), "saved": save}


def save_filter_results(role: str, preview_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = read_manifest(role)
    by_hash = {row.sha256: row for row in rows if row.sha256}
    now = now_tag()
    for preview in preview_rows:
        sha256 = str(preview.get("sha256", "")).strip()
        row = by_hash.get(sha256)
        if row is None:
            continue
        row.filter_status = str(preview.get("filter_status", row.filter_status) or row.filter_status)
        row.centroid_score = str(preview.get("centroid_score", row.centroid_score) or "")
        row.max_ref_score = str(preview.get("max_ref_score", row.max_ref_score) or "")
        row.median_ref_score = str(preview.get("median_ref_score", row.median_ref_score) or "")
        row.updated_at = now
    write_manifest(role, rows)
    return rows_as_dicts(rows)
