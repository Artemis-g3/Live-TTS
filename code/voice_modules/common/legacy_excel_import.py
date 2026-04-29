from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from voice_modules.common.audio_manifest import (
    import_audio_file,
    iter_audio_files,
    normalize_role,
    now_tag,
    read_manifest,
    role_library_path,
    rows_as_dicts,
    upsert_manifest_rows,
)
from voice_modules.common.workflow_db import (
    WORKFLOW_DB_FIELDS,
    build_library_keywords,
    ensure_workflow_db,
    read_workflow_rows,
    workflow_db_path,
    write_workflow_rows,
)


LEGACY_IMPORT_SOURCE = "legacy_excel_import"
REQUIRED_EXCEL_COLUMNS = ["索引", "语音文件", "语音文本", "自然语言描述", "情绪语气", "音频表达技巧"]
STANDARD_LIBRARY_COLUMNS = ["索引", "语音文件", "语音文本", "音频路径", "语音描述-音频理解模型", "关键词"]


@dataclass(frozen=True)
class ImportValidationError(Exception):
    message: str
    mismatch_count: int = 0
    sample_mismatches: tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.sample_mismatches:
            return self.message
        return f"{self.message} 示例: {'; '.join(self.sample_mismatches)}"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _load_excel_rows(excel_path: Path) -> list[dict[str, str]]:
    df = pd.read_excel(excel_path)
    missing = [column for column in REQUIRED_EXCEL_COLUMNS if column not in df.columns]
    if missing:
        raise ImportValidationError(f"Excel 缺少必要列: {missing}")
    rows: list[dict[str, str]] = []
    for raw in df.to_dict(orient="records"):
        rows.append({column: _clean_text(raw.get(column, "")) for column in REQUIRED_EXCEL_COLUMNS})
    return rows


def _validate_input_sets(audio_files: list[Path], excel_rows: list[dict[str, str]]) -> None:
    excel_names = [_clean_text(row["语音文件"]) for row in excel_rows]
    duplicate_names = sorted({name for name in excel_names if name and excel_names.count(name) > 1})
    if duplicate_names:
        raise ImportValidationError(
            "Excel 中存在重复语音文件名。",
            mismatch_count=len(duplicate_names),
            sample_mismatches=tuple(duplicate_names[:10]),
        )
    audio_names = [path.name for path in audio_files]
    if len(audio_names) != len(set(audio_names)):
        duplicates = sorted({name for name in audio_names if audio_names.count(name) > 1})
        raise ImportValidationError(
            "音频目录中存在重复文件名。",
            mismatch_count=len(duplicates),
            sample_mismatches=tuple(duplicates[:10]),
        )
    missing_in_excel = sorted(set(audio_names).difference(excel_names))
    missing_in_audio = sorted(set(excel_names).difference(audio_names))
    if missing_in_excel or missing_in_audio:
        samples = [f"音频缺 Excel: {name}" for name in missing_in_excel[:5]]
        samples.extend(f"Excel 缺音频: {name}" for name in missing_in_audio[:5])
        raise ImportValidationError(
            "Excel 与音频目录文件名集合不一致。",
            mismatch_count=len(missing_in_excel) + len(missing_in_audio),
            sample_mismatches=tuple(samples),
        )


def _import_manifest(role: str, audio_dir: Path) -> list[dict[str, Any]]:
    imported_rows = [
        import_audio_file(
            role=role,
            source=audio_path,
            selected_reference=False,
            filter_status="confirmed",
            source_label=LEGACY_IMPORT_SOURCE,
        )
        for audio_path in iter_audio_files(audio_dir)
    ]
    upsert_manifest_rows(role, imported_rows)
    manifest_rows = read_manifest(role)
    imported_names = {row.file_name for row in imported_rows}
    timestamp = now_tag()
    for row in manifest_rows:
        if row.file_name in imported_names:
            row.filter_status = "confirmed"
            row.selected_reference = False
            row.source = LEGACY_IMPORT_SOURCE
            row.updated_at = timestamp
    from voice_modules.common.audio_manifest import write_manifest

    write_manifest(role, manifest_rows)
    return rows_as_dicts(manifest_rows)


def _build_excel_lookup(excel_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["语音文件"]: row for row in excel_rows}


def _write_workflow_db(role: str, excel_rows: list[dict[str, str]]) -> tuple[Path, list[dict[str, str]]]:
    ensure_workflow_db(role)
    workflow_rows = read_workflow_rows(role, auto_sync=False)
    lookup = _build_excel_lookup(excel_rows)
    timestamp = now_tag()
    missing_rows: list[str] = []
    for row in workflow_rows:
        excel_row = lookup.get(row.get("file_name", ""))
        if excel_row is None:
            missing_rows.append(row.get("file_name", ""))
            continue
        tone = excel_row["情绪语气"]
        delivery = excel_row["音频表达技巧"]
        row["asr_status"] = "done"
        row["asr_model"] = LEGACY_IMPORT_SOURCE
        row["asr_language"] = "zh"
        row["asr_transcript"] = excel_row["语音文本"]
        row["asr_error"] = ""
        row["asr_updated_at"] = timestamp
        row["emotion_status"] = "done"
        row["emotion_model"] = LEGACY_IMPORT_SOURCE
        row["emotion_description"] = excel_row["自然语言描述"]
        row["emotion_tone"] = tone
        row["emotion_delivery"] = delivery
        row["emotion_error"] = ""
        row["emotion_updated_at"] = timestamp
        row["library_description"] = excel_row["自然语言描述"]
        row["library_keywords"] = build_library_keywords(tone, delivery)
        row["library_updated_at"] = timestamp
        row["db_updated_at"] = timestamp
    if missing_rows:
        raise ImportValidationError(
            "workflow_db 中存在无法匹配到 Excel 的文件名。",
            mismatch_count=len(missing_rows),
            sample_mismatches=tuple(sorted(missing_rows)[:10]),
        )
    normalized_rows: list[dict[str, Any]] = []
    for source_row in workflow_rows:
        row = {field: str(source_row.get(field, "") or "") for field in WORKFLOW_DB_FIELDS}
        normalized_rows.append(row)
    path = write_workflow_rows(role, normalized_rows)
    return path, normalized_rows


def _write_library_excel(role: str, excel_rows: list[dict[str, str]], workflow_rows: list[dict[str, str]]) -> Path:
    workflow_by_name = {row.get("file_name", ""): row for row in workflow_rows}
    records: list[dict[str, str]] = []
    for excel_row in excel_rows:
        file_name = excel_row["语音文件"]
        workflow_row = workflow_by_name.get(file_name)
        if workflow_row is None:
            raise ImportValidationError(
                "生成角色库 Excel 时发现缺失的 workflow 条目。",
                mismatch_count=1,
                sample_mismatches=(file_name,),
            )
        records.append(
            {
                "索引": excel_row["索引"],
                "语音文件": file_name,
                "语音文本": excel_row["语音文本"],
                "音频路径": workflow_row.get("audio_path", ""),
                "语音描述-音频理解模型": excel_row["自然语言描述"],
                "关键词": build_library_keywords(excel_row["情绪语气"], excel_row["音频表达技巧"]),
            }
        )
    path = role_library_path(role)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=STANDARD_LIBRARY_COLUMNS).to_excel(path, index=False)
    return path


def import_role_from_excel(role: str, audio_dir: Path, excel_path: Path) -> dict[str, Any]:
    normalized_role = normalize_role(role)
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise ImportValidationError(f"音频目录不存在: {audio_dir}")
    if not excel_path.exists() or not excel_path.is_file():
        raise ImportValidationError(f"Excel 文件不存在: {excel_path}")

    audio_files = iter_audio_files(audio_dir)
    if not audio_files:
        raise ImportValidationError(f"音频目录中没有可导入的音频: {audio_dir}")
    excel_rows = _load_excel_rows(excel_path)
    _validate_input_sets(audio_files, excel_rows)

    manifest_rows = _import_manifest(normalized_role, audio_dir)
    workflow_sync = ensure_workflow_db(normalized_role)
    if workflow_sync.row_count != len(excel_rows):
        raise ImportValidationError(
            "导入后 workflow_db 行数与 Excel 行数不一致。",
            mismatch_count=abs(workflow_sync.row_count - len(excel_rows)),
            sample_mismatches=(f"workflow={workflow_sync.row_count}, excel={len(excel_rows)}",),
        )
    workflow_db, workflow_rows = _write_workflow_db(normalized_role, excel_rows)
    library_excel = _write_library_excel(normalized_role, excel_rows, workflow_rows)

    return {
        "role": normalized_role,
        "audio_count": len(audio_files),
        "excel_row_count": len(excel_rows),
        "manifest_row_count": len(manifest_rows),
        "workflow_row_count": len(workflow_rows),
        "library_excel_path": str(library_excel.resolve()),
        "workflow_db_path": str(Path(workflow_db).resolve()),
        "mismatch_count": 0,
        "sample_mismatches": [],
    }
