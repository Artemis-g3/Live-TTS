import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from modelscope.pipelines import pipeline


SUPPORTED_EXTENSIONS = {".wav"}


@dataclass
class ReferenceStats:
    role: str
    reference_files: List[Path]
    active_reference_files: List[Path]
    centroid: np.ndarray
    embeddings: Dict[Path, np.ndarray]
    within_pair_scores: List[float]
    centroid_scores: Dict[Path, float]
    outliers: List[Path]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Filter target-speaker audio from mixed folders with CAM++."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=root / "speech_campplus_sv_zh-cn_16k-common",
        help="Local CAM++ model directory.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=root / "参考音频",
        help="Root directory containing per-role reference folders.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=root / "原始音频",
        help="Root directory containing per-role raw folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "筛选结果",
        help="Output directory for copied files and reports.",
    )
    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=None,
        help="Override auto-calibrated accept threshold.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=None,
        help="Override auto-calibrated review threshold.",
    )
    parser.add_argument(
        "--include-merged-ref",
        action="store_true",
        help="Include reference files with names ending in *_合并.wav.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of raw files processed per role for debugging.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate before feeding numpy arrays into the model.",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        default=None,
        help="Only process the specified role folder names.",
    )
    return parser.parse_args()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def read_audio(path: Path, sample_rate: int) -> np.ndarray:
    data, source_sr = sf.read(str(path), dtype="float32")
    if data.ndim == 2:
        data = data[:, 0]
    waveform = torch.from_numpy(data.astype("float32"))
    if source_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sr, sample_rate)
    return waveform.numpy().astype("float32")


def iter_audio_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def collect_reference_files(reference_dir: Path, include_merged_ref: bool) -> List[Path]:
    files = []
    for path in iter_audio_files(reference_dir):
        if not include_merged_ref and path.stem.endswith("_合并"):
            continue
        files.append(path)
    return files


def build_pipeline(model_dir: Path):
    return pipeline(task="speaker-verification", model=str(model_dir))


def extract_embedding(sv_pipeline, waveform: np.ndarray) -> np.ndarray:
    result = sv_pipeline([waveform], output_emb=True)
    emb = np.asarray(result["embs"][0], dtype="float32")
    return normalize_vector(emb)


def pairwise_scores(embeddings: Sequence[np.ndarray]) -> List[float]:
    scores: List[float] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            scores.append(cosine_similarity(embeddings[i], embeddings[j]))
    return scores


def build_reference_stats(
    role: str,
    reference_files: List[Path],
    sv_pipeline,
    sample_rate: int,
) -> ReferenceStats:
    if len(reference_files) < 2:
        raise ValueError(f"{role} 的参考音频不足，至少需要 2 条。")

    raw_embeddings: Dict[Path, np.ndarray] = {}
    for path in reference_files:
        raw_embeddings[path] = extract_embedding(sv_pipeline, read_audio(path, sample_rate))

    initial_centroid = normalize_vector(np.mean(np.stack(list(raw_embeddings.values())), axis=0))
    initial_scores = {
        path: cosine_similarity(embedding, initial_centroid)
        for path, embedding in raw_embeddings.items()
    }

    values = list(initial_scores.values())
    median_score = float(np.median(values))
    mad = float(np.median(np.abs(np.asarray(values) - median_score)))
    outlier_floor = median_score - max(0.03, 3.0 * mad)
    outliers = [path for path, score in initial_scores.items() if score < outlier_floor]

    active_reference_files = [path for path in reference_files if path not in outliers]
    if len(active_reference_files) < 2:
        active_reference_files = list(reference_files)

    embeddings = {path: raw_embeddings[path] for path in active_reference_files}
    centroid = normalize_vector(np.mean(np.stack(list(embeddings.values())), axis=0))
    centroid_scores = {
        path: cosine_similarity(embedding, centroid)
        for path, embedding in embeddings.items()
    }
    within_scores = pairwise_scores(list(embeddings.values()))

    return ReferenceStats(
        role=role,
        reference_files=reference_files,
        active_reference_files=active_reference_files,
        centroid=centroid,
        embeddings=embeddings,
        within_pair_scores=within_scores,
        centroid_scores=centroid_scores,
        outliers=outliers,
    )


def calibrate_thresholds(
    role_stats: ReferenceStats,
    all_stats: Dict[str, ReferenceStats],
    accept_override: Optional[float],
    review_override: Optional[float],
) -> Tuple[float, float, Dict[str, float]]:
    within = role_stats.within_pair_scores
    centroid_scores = list(role_stats.centroid_scores.values())

    cross_scores: List[float] = []
    for other_role, other_stats in all_stats.items():
        if other_role == role_stats.role:
            continue
        for emb in role_stats.embeddings.values():
            cross_scores.append(cosine_similarity(emb, other_stats.centroid))
        for emb in other_stats.embeddings.values():
            cross_scores.append(cosine_similarity(emb, role_stats.centroid))

    within_p25 = float(np.percentile(within, 25)) if within else min(centroid_scores)
    within_p10 = float(np.percentile(within, 10)) if within else min(centroid_scores)
    centroid_min = min(centroid_scores)
    cross_p95 = float(np.percentile(cross_scores, 95)) if cross_scores else -1.0
    cross_max = max(cross_scores) if cross_scores else -1.0

    auto_accept = max(
        cross_p95 + 0.03,
        min(within_p25, centroid_min - 0.01),
    )
    auto_review = max(
        cross_max + 0.02,
        min(within_p10, auto_accept - 0.05),
    )

    auto_accept = min(auto_accept, 0.98)
    auto_review = min(auto_review, auto_accept - 0.01)
    auto_review = max(auto_review, -1.0)

    accept_threshold = accept_override if accept_override is not None else auto_accept
    review_threshold = review_override if review_override is not None else auto_review

    if review_threshold >= accept_threshold:
        raise ValueError(
            f"{role_stats.role} 的 review_threshold 必须小于 accept_threshold。"
        )

    details = {
        "within_p25": within_p25,
        "within_p10": within_p10,
        "centroid_min": centroid_min,
        "cross_p95": cross_p95,
        "cross_max": cross_max,
        "auto_accept": auto_accept,
        "auto_review": auto_review,
        "accept_threshold": accept_threshold,
        "review_threshold": review_threshold,
    }
    return accept_threshold, review_threshold, details


def decide(
    centroid_score: float,
    max_ref_score: float,
    median_ref_score: float,
    accept_threshold: float,
    review_threshold: float,
) -> str:
    strong_match = min(centroid_score, max_ref_score)
    soft_match = min(max_ref_score, max(centroid_score, median_ref_score))
    if strong_match >= accept_threshold:
        return "confirmed"
    if soft_match >= review_threshold:
        return "review"
    return "reject"


def safe_copy(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if destination.exists():
        destination = destination_dir / f"{source.stem}_{hash(str(source)) & 0xffff:04x}{source.suffix}"
    shutil.copy2(source, destination)
    return destination


def process_role(
    role: str,
    raw_dir: Path,
    output_root: Path,
    sv_pipeline,
    sample_rate: int,
    reference_stats: ReferenceStats,
    accept_threshold: float,
    review_threshold: float,
    limit: Optional[int],
) -> List[Dict[str, object]]:
    confirmed_dir = output_root / role / "confirmed"
    review_dir = output_root / role / "review"

    rows: List[Dict[str, object]] = []
    raw_files = list(iter_audio_files(raw_dir))
    if limit is not None:
        raw_files = raw_files[:limit]

    reference_embeddings = list(reference_stats.embeddings.values())

    for index, path in enumerate(raw_files, start=1):
        waveform = read_audio(path, sample_rate)
        embedding = extract_embedding(sv_pipeline, waveform)
        centroid_score = cosine_similarity(embedding, reference_stats.centroid)
        ref_scores = [cosine_similarity(embedding, ref_emb) for ref_emb in reference_embeddings]
        max_ref_score = max(ref_scores)
        median_ref_score = float(np.median(ref_scores))
        decision = decide(
            centroid_score=centroid_score,
            max_ref_score=max_ref_score,
            median_ref_score=median_ref_score,
            accept_threshold=accept_threshold,
            review_threshold=review_threshold,
        )

        copied_to = ""
        if decision == "confirmed":
            copied_to = str(safe_copy(path, confirmed_dir))
        elif decision == "review":
            copied_to = str(safe_copy(path, review_dir))

        row = {
            "role": role,
            "source_path": str(path),
            "centroid_score": round(centroid_score, 6),
            "max_ref_score": round(max_ref_score, 6),
            "median_ref_score": round(median_ref_score, 6),
            "accept_threshold": round(accept_threshold, 6),
            "review_threshold": round(review_threshold, 6),
            "decision": decision,
            "copied_to": copied_to,
            "index": index,
        }
        rows.append(row)

        if index % 100 == 0 or index == len(raw_files):
            print(f"[{role}] processed {index}/{len(raw_files)}")

    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_reference_report(
    all_stats: Dict[str, ReferenceStats],
    threshold_details: Dict[str, Dict[str, float]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for role, stats in all_stats.items():
        for path in stats.reference_files:
            rows.append(
                {
                    "role": role,
                    "reference_path": str(path),
                    "centroid_score": round(stats.centroid_scores.get(path, float("nan")), 6),
                    "is_outlier": "yes" if path in stats.outliers else "no",
                    "used_in_model": "yes" if path in stats.active_reference_files else "no",
                    "accept_threshold": round(threshold_details[role]["accept_threshold"], 6),
                    "review_threshold": round(threshold_details[role]["review_threshold"], 6),
                }
            )
    return rows


def print_summary(
    all_rows: List[Dict[str, object]],
    threshold_details: Dict[str, Dict[str, float]],
    all_stats: Dict[str, ReferenceStats],
) -> None:
    print("\n=== Threshold Summary ===")
    for role, details in threshold_details.items():
        print(
            f"{role}: accept={details['accept_threshold']:.5f}, "
            f"review={details['review_threshold']:.5f}, "
            f"within_p25={details['within_p25']:.5f}, "
            f"cross_p95={details['cross_p95']:.5f}, "
            f"cross_max={details['cross_max']:.5f}"
        )
        if all_stats[role].outliers:
            outliers = ", ".join(path.name for path in all_stats[role].outliers)
            print(f"  outlier refs: {outliers}")
        used = ", ".join(path.name for path in all_stats[role].active_reference_files)
        print(f"  active refs: {used}")

    print("\n=== Decision Summary ===")
    by_role: Dict[str, Dict[str, int]] = {}
    for row in all_rows:
        bucket = by_role.setdefault(row["role"], {"confirmed": 0, "review": 0, "reject": 0})
        bucket[str(row["decision"])] += 1
    for role, counts in by_role.items():
        print(
            f"{role}: confirmed={counts['confirmed']}, "
            f"review={counts['review']}, reject={counts['reject']}"
        )


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    reference_root = args.reference_root.resolve()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()

    sv_pipeline = build_pipeline(model_dir)

    role_names = sorted(
        path.name for path in reference_root.iterdir() if path.is_dir() and (raw_root / path.name).is_dir()
    )
    if not role_names:
        raise ValueError("未找到同时存在于参考音频和原始音频中的角色目录。")
    if args.roles:
        requested = set(args.roles)
        role_names = [role for role in role_names if role in requested]
        if not role_names:
            raise ValueError("指定的 --roles 在参考音频和原始音频中都未找到。")

    all_stats: Dict[str, ReferenceStats] = {}
    for role in role_names:
        reference_files = collect_reference_files(reference_root / role, args.include_merged_ref)
        all_stats[role] = build_reference_stats(role, reference_files, sv_pipeline, args.sample_rate)

    threshold_details: Dict[str, Dict[str, float]] = {}
    resolved_thresholds: Dict[str, Tuple[float, float]] = {}
    for role in role_names:
        accept_threshold, review_threshold, details = calibrate_thresholds(
            role_stats=all_stats[role],
            all_stats=all_stats,
            accept_override=args.accept_threshold,
            review_override=args.review_threshold,
        )
        threshold_details[role] = details
        resolved_thresholds[role] = (accept_threshold, review_threshold)

    all_rows: List[Dict[str, object]] = []
    for role in role_names:
        accept_threshold, review_threshold = resolved_thresholds[role]
        all_rows.extend(
            process_role(
                role=role,
                raw_dir=raw_root / role,
                output_root=output_root,
                sv_pipeline=sv_pipeline,
                sample_rate=args.sample_rate,
                reference_stats=all_stats[role],
                accept_threshold=accept_threshold,
                review_threshold=review_threshold,
                limit=args.limit,
            )
        )

    write_csv(
        output_root / "report.csv",
        all_rows,
        [
            "role",
            "source_path",
            "centroid_score",
            "max_ref_score",
            "median_ref_score",
            "accept_threshold",
            "review_threshold",
            "decision",
            "copied_to",
            "index",
        ],
    )
    write_csv(
        output_root / "reference_report.csv",
        build_reference_report(all_stats, threshold_details),
        [
            "role",
            "reference_path",
            "centroid_score",
            "is_outlier",
            "used_in_model",
            "accept_threshold",
            "review_threshold",
        ],
    )

    print_summary(all_rows, threshold_details, all_stats)
    print(f"\nReports written to: {output_root}")


if __name__ == "__main__":
    main()
