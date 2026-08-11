from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voice_modules.common.audio_manifest import (
    ManifestRow,
    metadata_dir,
    normalize_role,
    now_tag,
    read_manifest,
    resolve_audio_path_for_row,
)


WORKFLOW_DB_FILENAME = "workflow_db.csv"
WORKFLOW_DB_FIELDS = [
    "role",
    "sha256",
    "file_name",
    "stored_name",
    "audio_path",
    "duration_seconds",
    "sample_rate",
    "source",
    "filter_status",
    "selected_reference",
    "manifest_created_at",
    "manifest_updated_at",
    "db_created_at",
    "db_updated_at",
    "asr_status",
    "asr_model",
    "asr_language",
    "asr_original_text",
    "asr_transcript",
    "asr_error",
    "asr_updated_at",
    "translation_status",
    "translation_model",
    "translation_error",
    "translation_updated_at",
    "emotion_status",
    "emotion_model",
    "emotion_description",
    "emotion_tone",
    "emotion_delivery",
    "emotion_error",
    "emotion_updated_at",
    "library_description",
    "library_keywords",
    "library_updated_at",
]

IDX = "索引"
AUDIO_FILE = "语音文件"
TRANSCRIPT = "语音文本"
AUDIO_PATH = "音频路径"
DESCRIPTION = "自然语言描述"
TONE = "情绪语气"
DELIVERY = "音频表达技巧"
LIB_DESC = "语音描述-音频理解模型"
KEYWORDS = "关键词"

ASR_FILE = "语音文件"
ASR_LANGUAGE = "原始语言"
ASR_ORIGINAL_TEXT = "原始文本"
ASR_CHINESE_TEXT = "中文文本"
ASR_ERROR = "ASR错误"
TRANSLATION_ERROR = "翻译错误"


@dataclass(frozen=True)
class WorkflowSyncResult:
    role: str
    path: Path
    row_count: int


def workflow_db_path(role: str) -> Path:
    return metadata_dir(role) / WORKFLOW_DB_FILENAME


def _empty_row() -> dict[str, str]:
    return {field: "" for field in WORKFLOW_DB_FIELDS}


def build_library_keywords(tone: str, delivery: str) -> str:
    parts = [str(value).strip() for value in [tone, delivery] if str(value).strip()]
    return "，".join(parts)


def _manifest_to_workflow_row(row: ManifestRow) -> dict[str, str]:
    now = now_tag()
    data = _empty_row()
    data.update(
        {
            "role": row.role,
            "sha256": row.sha256,
            "file_name": row.file_name,
            "stored_name": row.stored_name,
            "audio_path": row.audio_path,
            "duration_seconds": f"{row.duration_seconds:.3f}",
            "sample_rate": str(row.sample_rate),
            "source": row.source,
            "filter_status": row.filter_status,
            "selected_reference": "true" if row.selected_reference else "false",
            "manifest_created_at": row.created_at,
            "manifest_updated_at": row.updated_at,
            "db_created_at": now,
            "db_updated_at": now,
        }
    )
    return data


def _refresh_base_fields(target: dict[str, str], manifest_row: ManifestRow) -> None:
    target.update(
        {
            "role": manifest_row.role,
            "sha256": manifest_row.sha256,
            "file_name": manifest_row.file_name,
            "stored_name": manifest_row.stored_name,
            "audio_path": manifest_row.audio_path,
            "duration_seconds": f"{manifest_row.duration_seconds:.3f}",
            "sample_rate": str(manifest_row.sample_rate),
            "source": manifest_row.source,
            "filter_status": manifest_row.filter_status,
            "selected_reference": "true" if manifest_row.selected_reference else "false",
            "manifest_created_at": manifest_row.created_at,
            "manifest_updated_at": manifest_row.updated_at,
            "db_updated_at": now_tag(),
        }
    )


def _rows_equal(left: list[dict[str, str]], right: list[dict[str, str]]) -> bool:
    if len(left) != len(right):
        return False
    left_by_hash = workflow_rows_by_sha(left)
    right_by_hash = workflow_rows_by_sha(right)
    if len(left_by_hash) != len(right_by_hash):
        return False
    ignored_fields = {"db_updated_at"}
    for sha, left_row in left_by_hash.items():
        right_row = right_by_hash.get(sha)
        if right_row is None:
            return False
        if any(
            left_row.get(field) != right_row.get(field)
            for field in WORKFLOW_DB_FIELDS
            if field not in ignored_fields
        ):
            return False
    return True


def ensure_workflow_db(role: str) -> WorkflowSyncResult:
    normalized = normalize_role(role)
    existing_rows = read_workflow_rows(normalized, auto_sync=False)
    by_hash = {row.get("sha256", ""): row for row in existing_rows if row.get("sha256")}
    manifest_rows = [row for row in read_manifest(normalized) if row.sha256]
    synced_rows: list[dict[str, str]] = []
    synced_hashes: set[str] = set()
    for manifest_row in manifest_rows:
        existing = by_hash.get(manifest_row.sha256)
        if existing is None:
            synced_rows.append(_manifest_to_workflow_row(manifest_row))
        else:
            refreshed = _empty_row()
            refreshed.update(existing)
            _refresh_base_fields(refreshed, manifest_row)
            synced_rows.append(refreshed)
        synced_hashes.add(manifest_row.sha256)
    for sha, row in by_hash.items():
        if sha not in synced_hashes:
            synced_rows.append(row)
    if not _rows_equal(existing_rows, synced_rows):
        write_workflow_rows(normalized, synced_rows)
    return WorkflowSyncResult(role=normalized, path=workflow_db_path(normalized), row_count=len(synced_rows))


def read_workflow_rows(role: str, *, auto_sync: bool = True) -> list[dict[str, str]]:
    normalized = normalize_role(role)
    path = workflow_db_path(normalized)
    if not path.exists():
        if auto_sync:
            ensure_workflow_db(normalized)
        else:
            return []
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows: list[dict[str, str]] = []
        for raw_row in csv.DictReader(handle):
            row = _empty_row()
            row.update({field: str(raw_row.get(field, "") or "") for field in WORKFLOW_DB_FIELDS})
            rows.append(row)
        return rows


def write_workflow_rows(role: str, rows: list[dict[str, Any]]) -> Path:
    normalized = normalize_role(role)
    path = workflow_db_path(normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WORKFLOW_DB_FIELDS)
        writer.writeheader()
        for source_row in rows:
            row = _empty_row()
            row.update({field: str(source_row.get(field, "") or "") for field in WORKFLOW_DB_FIELDS})
            writer.writerow(row)
    return path


def workflow_rows_by_sha(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["sha256"]: row for row in rows if row.get("sha256")}


def _sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (row.get("file_name", ""), row.get("sha256", "")))


def load_asr_rows(role: str) -> list[dict[str, str]]:
    rows = _sort_rows(read_workflow_rows(role))
    return [
        {
            "sha256": row.get("sha256", ""),
            "role": row.get("role", ""),
            "file_name": row.get("file_name", ""),
            "file_path": str(
                resolve_audio_path_for_row(
                    role=row.get("role", "") or role,
                    audio_path=row.get("audio_path", ""),
                    stored_name=row.get("stored_name", ""),
                    file_name=row.get("file_name", ""),
                )
            ),
            "model": row.get("asr_model", ""),
            "translation_model": row.get("translation_model", ""),
            ASR_FILE: row.get("file_name", ""),
            ASR_LANGUAGE: row.get("asr_language", ""),
            ASR_ORIGINAL_TEXT: row.get("asr_original_text", ""),
            ASR_CHINESE_TEXT: row.get("asr_transcript", ""),
            ASR_ERROR: row.get("asr_error", ""),
            TRANSLATION_ERROR: row.get("translation_error", ""),
        }
        for row in rows
    ]


def _payload_text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(payload.get(key, "") or "").strip()
        if value:
            return value
    return ""


def save_asr_view_rows(role: str, rows: list[dict[str, Any]]) -> Path:
    existing_rows = read_workflow_rows(role)
    by_hash = workflow_rows_by_sha(existing_rows)
    now = now_tag()
    for payload in rows:
        sha256 = str(payload.get("sha256", "") or "").strip()
        target = by_hash.get(sha256)
        if target is None:
            continue
        language = _payload_text(payload, ASR_LANGUAGE, "language")
        original_text = _payload_text(payload, ASR_ORIGINAL_TEXT, "original_text")
        chinese_text = str(payload.get(ASR_CHINESE_TEXT, payload.get("transcript", "")) or "").strip()
        asr_error = _payload_text(payload, ASR_ERROR, "error")
        translation_error = str(payload.get(TRANSLATION_ERROR, payload.get("translation_error", "")) or "").strip()
        translation_model = str(payload.get("translation_model", "") or "").strip()
        if language == "zh" and original_text and not chinese_text:
            chinese_text = original_text
        if language == "zh" and original_text and not translation_model:
            translation_model = "direct_copy"

        target["asr_model"] = str(payload.get("model", "") or "").strip()
        target["asr_language"] = language
        target["asr_original_text"] = original_text
        target["asr_transcript"] = chinese_text
        target["asr_error"] = asr_error
        target["asr_status"] = "done" if original_text else ("error" if asr_error else "")
        target["asr_updated_at"] = now if original_text or asr_error else ""
        target["translation_model"] = translation_model
        target["translation_error"] = translation_error
        target["translation_status"] = (
            "done"
            if chinese_text
            else ("error" if translation_error else "")
        )
        target["translation_updated_at"] = now if chinese_text or translation_error else ""
        target["db_updated_at"] = now
    return write_workflow_rows(role, _sort_rows(existing_rows))


def load_emotion_rows(role: str) -> list[dict[str, str]]:
    rows = [
        row
        for row in _sort_rows(read_workflow_rows(role))
        if row.get("asr_original_text") or row.get("asr_transcript") or row.get("emotion_status")
    ]
    result: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        result.append(
            {
                "sha256": row.get("sha256", ""),
                "model": row.get("emotion_model", ""),
                "error": row.get("emotion_error", ""),
                IDX: str(index),
                AUDIO_FILE: row.get("file_name", ""),
                TRANSCRIPT: row.get("asr_transcript", ""),
                AUDIO_PATH: str(
                    resolve_audio_path_for_row(
                        role=row.get("role", "") or role,
                        audio_path=row.get("audio_path", ""),
                        stored_name=row.get("stored_name", ""),
                        file_name=row.get("file_name", ""),
                    )
                ),
                DESCRIPTION: row.get("emotion_description", ""),
                TONE: row.get("emotion_tone", ""),
                DELIVERY: row.get("emotion_delivery", ""),
                KEYWORDS: row.get("library_keywords", "") or build_library_keywords(row.get("emotion_tone", ""), row.get("emotion_delivery", "")),
            }
        )
    return result


def save_emotion_view_rows(role: str, rows: list[dict[str, Any]], model: str = "") -> Path:
    existing_rows = read_workflow_rows(role)
    by_hash = workflow_rows_by_sha(existing_rows)
    now = now_tag()
    for payload in rows:
        sha256 = str(payload.get("sha256", "") or "").strip()
        target = by_hash.get(sha256)
        if target is None:
            continue
        description = str(payload.get(DESCRIPTION, "") or "").strip()
        tone = str(payload.get(TONE, "") or "").strip()
        delivery = str(payload.get(DELIVERY, "") or "").strip()
        keywords = str(payload.get(KEYWORDS, "") or "").strip()
        error = str(payload.get("error", "") or "").strip()
        target["emotion_model"] = model.strip() or target.get("emotion_model", "")
        target["emotion_description"] = description
        target["emotion_tone"] = tone
        target["emotion_delivery"] = delivery
        target["emotion_error"] = error
        target["emotion_status"] = "done" if description or tone or delivery else ("error" if error else "")
        target["emotion_updated_at"] = now if target["emotion_status"] else ""
        target["library_description"] = description
        target["library_keywords"] = keywords
        if target.get("library_description", "").strip() or target.get("library_keywords", "").strip():
            target["library_updated_at"] = now
        target["db_updated_at"] = now
    return write_workflow_rows(role, _sort_rows(existing_rows))


def refresh_library_fields(role: str) -> tuple[Path, int]:
    rows = read_workflow_rows(role)
    updated = 0
    now = now_tag()
    for row in rows:
        changed = False
        if row.get("emotion_description", "").strip() and not row.get("library_description", "").strip():
            row["library_description"] = row["emotion_description"].strip()
            changed = True
        if not row.get("library_keywords", "").strip():
            keywords = build_library_keywords(row.get("emotion_tone", ""), row.get("emotion_delivery", ""))
            if keywords:
                row["library_keywords"] = keywords
                changed = True
        if changed:
            row["library_updated_at"] = now
            row["db_updated_at"] = now
            updated += 1
    return write_workflow_rows(role, _sort_rows(rows)), updated


def load_library_rows(role: str, limit: int = 0) -> list[dict[str, str]]:
    existing_workflow_rows = [
        row
        for row in _sort_rows(read_workflow_rows(role))
        if row.get("asr_transcript", "").strip()
        or row.get("library_description", "").strip()
        or row.get("library_keywords", "").strip()
    ]
    rows = load_emotion_rows(role)
    existing_rows = workflow_rows_by_sha(existing_workflow_rows)
    result: list[dict[str, str]] = []
    for row in rows:
        source = existing_rows.get(row.get("sha256", ""), {})
        if not source:
            continue
        result.append(
            {
                "sha256": row.get("sha256", ""),
                IDX: row.get(IDX, ""),
                AUDIO_FILE: row.get(AUDIO_FILE, ""),
                TRANSCRIPT: row.get(TRANSCRIPT, ""),
                AUDIO_PATH: row.get(AUDIO_PATH, ""),
                LIB_DESC: source.get("library_description", ""),
                KEYWORDS: source.get("library_keywords", ""),
            }
        )
    if limit > 0:
        return result[:limit]
    return result


def save_library_rows(role: str, rows: list[dict[str, Any]]) -> Path:
    existing_rows = read_workflow_rows(role)
    by_hash = workflow_rows_by_sha(existing_rows)
    now = now_tag()
    for payload in rows:
        sha256 = str(payload.get("sha256", "") or "").strip()
        target = by_hash.get(sha256)
        if target is None:
            continue
        target["library_description"] = str(payload.get(LIB_DESC, "") or "").strip()
        target["library_keywords"] = str(payload.get(KEYWORDS, "") or "").strip()
        target["library_updated_at"] = now
        target["db_updated_at"] = now
    return write_workflow_rows(role, _sort_rows(existing_rows))
