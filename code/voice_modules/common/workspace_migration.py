from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from voice_modules.common.audio_manifest import (
    PROJECT_ROOT,
    WORKSPACE_ROOT,
    asr_path,
    emotion_path,
    file_sha256,
    import_audio_file,
    iter_audio_files,
    library_dir,
    normalize_role,
    role_library_path,
    rows_as_dicts,
    read_manifest,
    upsert_manifest_rows,
)


def discover_legacy_roles() -> list[str]:
    roles: set[str] = set()
    for folder in ["1.原始音频", "1.参考音频", "1.筛选音频", "4.整理音频"]:
        root = PROJECT_ROOT / folder
        if not root.exists():
            continue
        for item in root.iterdir():
            if item.is_dir() and not item.name.startswith("__"):
                roles.add(normalize_role(item.name))
    for pattern_root in [PROJECT_ROOT / "2.语音识别", PROJECT_ROOT / "4.整理音频"]:
        if not pattern_root.exists():
            continue
        for csv_path in pattern_root.rglob("*.csv"):
            name = csv_path.stem
            if name.endswith("_语音识别结果"):
                roles.add(normalize_role(name[: -len("_语音识别结果")]))
            if name.endswith("_情感标定"):
                roles.add(normalize_role(name[: -len("_情感标定")]))
    return sorted(roles)


def migrate_audio_for_role(role: str) -> list[dict[str, Any]]:
    imported = []
    sources = [
        ("1.原始音频", False, "unprocessed", "legacy_raw"),
        ("1.参考音频", True, "unprocessed", "legacy_reference"),
        (str(Path("1.筛选音频") / role / "confirmed"), False, "confirmed", "legacy_confirmed"),
        (str(Path("1.筛选音频") / role / "review"), False, "review", "legacy_review"),
        ("4.整理音频", False, "confirmed", "legacy_sorted"),
    ]
    for root_name, is_reference, status, label in sources:
        root = PROJECT_ROOT / root_name
        role_dir = root if root.name in {"confirmed", "review"} else root / role
        for audio_path in iter_audio_files(role_dir):
            imported.append(
                import_audio_file(
                    role=role,
                    source=audio_path,
                    selected_reference=is_reference,
                    filter_status=status,
                    source_label=label,
                )
            )
    rows = upsert_manifest_rows(role, imported)
    return rows_as_dicts(rows)


def copy_csv_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def workspace_audio_lookup(role: str) -> tuple[dict[str, str], dict[str, str]]:
    by_name = {}
    by_hash = {}
    for row in read_manifest(role):
        by_name.setdefault(row.file_name, row.audio_path)
        by_name.setdefault(row.stored_name, row.audio_path)
        if row.sha256:
            by_hash.setdefault(row.sha256, row.audio_path)
    return by_name, by_hash


def resolve_workspace_audio_path(original_path: str, file_name: str, by_name: dict[str, str], by_hash: dict[str, str]) -> str:
    if file_name and file_name in by_name:
        return by_name[file_name]
    source = Path(str(original_path or ""))
    if source.exists():
        try:
            return by_hash.get(file_sha256(source), str(source))
        except Exception:
            return str(source)
    return str(original_path or "")


def migrate_asr_for_role(role: str) -> bool:
    candidates = [
        PROJECT_ROOT / "2.语音识别" / role / f"{role}_语音识别结果.csv",
        PROJECT_ROOT / "2.语音识别" / "筛选音频识别结果.csv",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if not row.get("role") or row.get("role") == role]
        if not rows:
            continue
        by_name, by_hash = workspace_audio_lookup(role)
        for row in rows:
            if "file_path" in row:
                row["file_path"] = resolve_workspace_audio_path(
                    row.get("file_path", ""),
                    row.get("file_name", ""),
                    by_name,
                    by_hash,
                )
        destination = asr_path(role)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with destination.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return True
    return False


def migrate_emotion_for_role(role: str) -> bool:
    source = PROJECT_ROOT / "4.整理音频" / f"{role}_情感标定.csv"
    if not source.exists():
        return False
    df = pd.read_csv(source, encoding="utf-8-sig")
    by_name, by_hash = workspace_audio_lookup(role)
    if "语音文件" in df.columns and "音频路径" in df.columns:
        df["音频路径"] = df.apply(
            lambda row: resolve_workspace_audio_path(
                row.get("音频路径", ""),
                str(row.get("语音文件", "")),
                by_name,
                by_hash,
            ),
            axis=1,
        )
    destination = emotion_path(role)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False, encoding="utf-8-sig")
    return True


def migrate_library_for_role(role: str) -> bool:
    destination = role_library_path(role)
    emotion_csv = emotion_path(role)
    if not emotion_csv.exists():
        source = PROJECT_ROOT / "5.双层情感检索配音-pro" / "角色数据" / role / "情绪关键词.xlsx"
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return True
        return False
    df = pd.read_csv(emotion_csv, encoding="utf-8-sig")
    required = {"索引", "语音文件", "语音文本", "音频路径", "自然语言描述", "情绪语气", "音频表达技巧"}
    if required.difference(df.columns):
        return False
    out = pd.DataFrame(
        {
            "索引": df["索引"],
            "语音文件": df["语音文件"],
            "语音文本": df["语音文本"],
            "音频路径": df["音频路径"],
            "语音描述-音频理解模型": df["自然语言描述"],
            "关键词": df[["情绪语气", "音频表达技巧"]].fillna("").agg("，".join, axis=1),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(destination, index=False)
    return True


def migrate_workspace() -> dict[str, Any]:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    roles = discover_legacy_roles()
    results = []
    for role in roles:
        manifest_rows = migrate_audio_for_role(role)
        results.append(
            {
                "role": role,
                "manifest_rows": len(manifest_rows),
                "asr": migrate_asr_for_role(role),
                "emotion": migrate_emotion_for_role(role),
                "library": migrate_library_for_role(role),
            }
        )
    return {"workspace_root": str(WORKSPACE_ROOT), "roles": results}
