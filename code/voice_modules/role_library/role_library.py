from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voice_modules.common.audio_manifest import normalize_role, output_dir, raw_audio_dir
from voice_modules.common.workflow_db import (
    AUDIO_FILE,
    AUDIO_PATH,
    IDX,
    KEYWORDS,
    LIB_DESC,
    TRANSCRIPT,
    load_library_rows,
    refresh_library_fields,
    save_library_rows,
    workflow_db_path,
)


LIBRARY_COLUMNS = ["sha256", IDX, AUDIO_FILE, TRANSCRIPT, AUDIO_PATH, LIB_DESC, KEYWORDS]


@dataclass(frozen=True)
class RolePaths:
    role: str
    audio_dir: Path
    csv_path: Path
    excel_path: Path
    output_dir: Path


@dataclass(frozen=True)
class LibraryRefreshResult:
    role: str
    workflow_db_path: Path
    updated_rows: int


def resolve_role_paths(role: str) -> RolePaths:
    normalized = normalize_role(role)
    db_path = workflow_db_path(normalized).resolve()
    return RolePaths(
        role=normalized,
        audio_dir=raw_audio_dir(normalized).resolve(),
        csv_path=db_path,
        excel_path=db_path,
        output_dir=output_dir(normalized).resolve(),
    )


def prepare_role_excel(role: str, *, force: bool = False) -> LibraryRefreshResult:
    del force
    normalized = normalize_role(role)
    db_path, updated_rows = refresh_library_fields(normalized)
    return LibraryRefreshResult(role=normalized, workflow_db_path=db_path, updated_rows=updated_rows)


def preview_role_library(role: str, limit: int = 0) -> list[dict[str, Any]]:
    return load_library_rows(role, limit=limit)


def save_role_library(role: str, rows: list[dict[str, Any]]) -> LibraryRefreshResult:
    if not rows:
        raise ValueError("没有可保存的角色库条目。")
    missing = set(LIBRARY_COLUMNS).difference(rows[0].keys())
    if missing:
        raise ValueError(f"保存数据缺少字段: {sorted(missing)}")
    path = save_library_rows(role, rows)
    return LibraryRefreshResult(role=normalize_role(role), workflow_db_path=path, updated_rows=len(rows))
