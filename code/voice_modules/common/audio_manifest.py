from __future__ import annotations

import csv
import hashlib
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from GUI.config import PROJECT_ROOT


WORKSPACE_ROOT = PROJECT_ROOT / "workspace"
INPUT_ROOT = PROJECT_ROOT / "input_audio"
ROLES_ROOT = WORKSPACE_ROOT / "roles"
TRASH_ROOT = WORKSPACE_ROOT / "trash"

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

ROLE_ALIASES = {
    "moning": "莫宁",
    "morning": "莫宁",
    "m": "莫宁",
    "1": "莫宁",
    "yuno": "尤诺",
    "younuo": "尤诺",
    "y": "尤诺",
    "2": "尤诺",
}

MANIFEST_FIELDS = [
    "role",
    "file_name",
    "stored_name",
    "audio_path",
    "duration_seconds",
    "sample_rate",
    "sha256",
    "selected_reference",
    "filter_status",
    "centroid_score",
    "max_ref_score",
    "median_ref_score",
    "source",
    "created_at",
    "updated_at",
]


@dataclass
class ManifestRow:
    role: str
    file_name: str
    stored_name: str
    audio_path: str
    duration_seconds: float = 0.0
    sample_rate: int = 0
    sha256: str = ""
    selected_reference: bool = False
    filter_status: str = "unprocessed"
    centroid_score: str = ""
    max_ref_score: str = ""
    median_ref_score: str = ""
    source: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ManifestRow":
        data = {field_name: row.get(field_name, "") for field_name in MANIFEST_FIELDS}
        data["duration_seconds"] = float(data["duration_seconds"] or 0)
        data["sample_rate"] = int(float(data["sample_rate"] or 0))
        data["selected_reference"] = str(data["selected_reference"]).lower() in {"1", "true", "yes", "是"}
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "file_name": self.file_name,
            "stored_name": self.stored_name,
            "audio_path": self.audio_path,
            "duration_seconds": f"{self.duration_seconds:.3f}",
            "sample_rate": self.sample_rate,
            "sha256": self.sha256,
            "selected_reference": "true" if self.selected_reference else "false",
            "filter_status": self.filter_status,
            "centroid_score": self.centroid_score,
            "max_ref_score": self.max_ref_score,
            "median_ref_score": self.median_ref_score,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def normalize_role(role: str) -> str:
    value = role.strip()
    if not value:
        raise ValueError("角色名不能为空。")
    return ROLE_ALIASES.get(value.lower(), value)


def role_root(role: str) -> Path:
    return ROLES_ROOT / normalize_role(role)


def raw_audio_dir(role: str) -> Path:
    return role_root(role) / "audio" / "raw"


def metadata_dir(role: str) -> Path:
    return role_root(role) / "metadata"


def library_dir(role: str) -> Path:
    return role_root(role) / "library"


def output_dir(role: str) -> Path:
    return role_root(role) / "output"


def project_relative_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def resolve_project_path(path_text: str) -> Path:
    path = Path(str(path_text or "")).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_audio_path_for_row(
    *,
    role: str,
    audio_path: str = "",
    stored_name: str = "",
    file_name: str = "",
) -> Path:
    normalized = normalize_role(role)
    candidates: list[Path] = []
    if str(audio_path or "").strip():
        candidates.append(resolve_project_path(audio_path))
    for name in [stored_name, file_name]:
        clean_name = Path(str(name or "")).name
        if clean_name:
            candidates.append(raw_audio_dir(normalized) / clean_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if candidates:
        return candidates[0].resolve()
    return raw_audio_dir(normalized).resolve()


def manifest_path(role: str) -> Path:
    return metadata_dir(role) / "audio_manifest.csv"


def asr_path(role: str) -> Path:
    return metadata_dir(role) / "asr_results.csv"


def emotion_path(role: str) -> Path:
    return metadata_dir(role) / "emotion_results.csv"


def role_library_path(role: str) -> Path:
    return library_dir(role) / "情绪关键词.xlsx"


def ensure_role_dirs(role: str) -> None:
    for path in [raw_audio_dir(role), metadata_dir(role), library_dir(role), output_dir(role)]:
        path.mkdir(parents=True, exist_ok=True)


def discover_workspace_roles() -> list[str]:
    if not ROLES_ROOT.exists():
        return []
    return sorted(path.name for path in ROLES_ROOT.iterdir() if path.is_dir())


def iter_audio_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_info(path: Path) -> tuple[float, int]:
    try:
        info = sf.info(str(path))
        duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
        return duration, int(info.samplerate or 0)
    except Exception:
        return 0.0, 0


def read_manifest(role: str) -> list[ManifestRow]:
    path = manifest_path(role)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [ManifestRow.from_dict(row) for row in csv.DictReader(handle)]


def write_manifest(role: str, rows: list[ManifestRow]) -> None:
    ensure_role_dirs(role)
    path = manifest_path(role)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def upsert_manifest_rows(role: str, new_rows: list[ManifestRow]) -> list[ManifestRow]:
    rows = read_manifest(role)
    by_hash = {row.sha256: row for row in rows if row.sha256}
    for row in new_rows:
        existing = by_hash.get(row.sha256)
        if existing:
            if row.selected_reference:
                existing.selected_reference = True
            if existing.filter_status in {"", "unprocessed"} and row.filter_status:
                existing.filter_status = row.filter_status
            existing.source = existing.source or row.source
            existing.updated_at = now_tag()
        else:
            rows.append(row)
            by_hash[row.sha256] = row
    write_manifest(role, rows)
    return rows


def now_tag() -> str:
    return datetime.now().isoformat(timespec="seconds")


def unique_stored_name(source: Path, sha256: str) -> str:
    safe_stem = source.stem[:80] or "audio"
    return f"{safe_stem}_{sha256[:10]}{source.suffix.lower()}"


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        destination.hardlink_to(source)
    except Exception:
        shutil.copy2(source, destination)


def import_audio_file(
    *,
    role: str,
    source: Path,
    selected_reference: bool = False,
    filter_status: str = "unprocessed",
    source_label: str = "",
) -> ManifestRow:
    normalized_role = normalize_role(role)
    ensure_role_dirs(normalized_role)
    digest = file_sha256(source)
    stored_name = unique_stored_name(source, digest)
    destination = raw_audio_dir(normalized_role) / stored_name
    link_or_copy(source, destination)
    duration, sample_rate = audio_info(destination)
    now = now_tag()
    return ManifestRow(
        role=normalized_role,
        file_name=source.name,
        stored_name=stored_name,
        audio_path=project_relative_path(destination),
        duration_seconds=duration,
        sample_rate=sample_rate,
        sha256=digest,
        selected_reference=selected_reference,
        filter_status=filter_status or "unprocessed",
        source=source_label,
        created_at=now,
        updated_at=now,
    )


def set_manifest_status(role: str, sha256_values: list[str], **updates: Any) -> list[ManifestRow]:
    rows = read_manifest(role)
    wanted = set(sha256_values)
    for row in rows:
        if row.sha256 not in wanted:
            continue
        for key, value in updates.items():
            if hasattr(row, key):
                setattr(row, key, value)
        row.updated_at = now_tag()
    write_manifest(role, rows)
    return rows


def move_rows_to_trash(role: str, sha256_values: list[str]) -> list[ManifestRow]:
    rows = read_manifest(role)
    wanted = set(sha256_values)
    trash_dir = TRASH_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S") / normalize_role(role)
    trash_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row.sha256 not in wanted:
            continue
        source = resolve_audio_path_for_row(
            role=row.role,
            audio_path=row.audio_path,
            stored_name=row.stored_name,
            file_name=row.file_name,
        )
        if source.exists():
            target = trash_dir / source.name
            shutil.move(str(source), str(target))
            row.audio_path = project_relative_path(target)
        row.filter_status = "deleted"
        row.updated_at = now_tag()
    write_manifest(role, rows)
    return rows


def rows_as_dicts(rows: list[ManifestRow]) -> list[dict[str, Any]]:
    return [row.to_dict() for row in rows]
