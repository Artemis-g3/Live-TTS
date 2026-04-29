from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voice_modules.common.audio_manifest import ManifestRow, metadata_dir, normalize_role, now_tag, read_manifest, resolve_audio_path_for_row


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
    "asr_transcript",
    "asr_error",
    "asr_updated_at",
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

IDX = "\u7d22\u5f15"
AUDIO_FILE = "\u8bed\u97f3\u6587\u4ef6"
TRANSCRIPT = "\u8bed\u97f3\u6587\u672c"
AUDIO_PATH = "\u97f3\u9891\u8def\u5f84"
DESCRIPTION = "\u81ea\u7136\u8bed\u8a00\u63cf\u8ff0"
TONE = "\u60c5\u7eea\u8bed\u6c14"
DELIVERY = "\u97f3\u9891\u8868\u8fbe\u6280\u5de7"
LIB_DESC = "\u8bed\u97f3\u63cf\u8ff0-\u97f3\u9891\u7406\u89e3\u6a21\u578b"
KEYWORDS = "\u5173\u952e\u8bcd"


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
    return "\uff0c".join(parts)


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


def ensure_workflow_db(role: str) -> WorkflowSyncResult:
    normalized = normalize_role(role)
    existing_rows = read_workflow_rows(normalized, auto_sync=False)
    by_hash = {row.get("sha256", ""): row for row in existing_rows if row.get("sha256")}
    confirmed_rows = [row for row in read_manifest(normalized) if row.filter_status == "confirmed" and row.sha256]
    synced_rows: list[dict[str, str]] = []
    for manifest_row in confirmed_rows:
        existing = by_hash.get(manifest_row.sha256)
        if existing is None:
            synced_rows.append(_manifest_to_workflow_row(manifest_row))
            continue
        refreshed = _empty_row()
        refreshed.update(existing)
        _refresh_base_fields(refreshed, manifest_row)
        synced_rows.append(refreshed)
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
            "language": row.get("asr_language", ""),
            "transcript": row.get("asr_transcript", ""),
            "error": row.get("asr_error", ""),
        }
        for row in rows
    ]


def save_asr_view_rows(role: str, rows: list[dict[str, Any]]) -> Path:
    existing_rows = read_workflow_rows(role)
    by_hash = workflow_rows_by_sha(existing_rows)
    now = now_tag()
    for payload in rows:
        sha256 = str(payload.get("sha256", "") or "").strip()
        target = by_hash.get(sha256)
        if target is None:
            continue
        transcript = str(payload.get("transcript", "") or "").strip()
        error = str(payload.get("error", "") or "").strip()
        target["asr_model"] = str(payload.get("model", "") or "").strip()
        target["asr_language"] = str(payload.get("language", "") or "").strip()
        target["asr_transcript"] = transcript
        target["asr_error"] = error
        target["asr_status"] = "done" if transcript else ("error" if error else "")
        target["asr_updated_at"] = now if transcript or error else ""
        target["db_updated_at"] = now
    return write_workflow_rows(role, _sort_rows(existing_rows))


def load_emotion_rows(role: str) -> list[dict[str, str]]:
    rows = [row for row in _sort_rows(read_workflow_rows(role)) if row.get("asr_transcript") or row.get("emotion_status")]
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
    rows = load_emotion_rows(role)
    existing_rows = workflow_rows_by_sha(read_workflow_rows(role))
    result: list[dict[str, str]] = []
    for row in rows:
        source = existing_rows.get(row.get("sha256", ""), {})
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
