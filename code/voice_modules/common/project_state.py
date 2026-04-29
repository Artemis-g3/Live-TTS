from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from voice_modules.common.audio_manifest import (
    INPUT_ROOT,
    discover_workspace_roles,
    manifest_path,
    normalize_role,
    read_manifest,
)
from voice_modules.dubbing.output_overview import latest_tts_run_info
from voice_modules.common.workflow_db import read_workflow_rows


@dataclass
class RoleState:
    role: str
    input_audio_count: int
    raw_audio_count: int
    reference_audio_count: int
    confirmed_audio_count: int
    review_audio_count: int
    excluded_audio_count: int
    asr_rows: int
    emotion_rows: int
    has_role_excel: bool
    tts_run_count: int
    latest_tts_run: str
    latest_synthesized_audio: str


def count_stage_rows(rows: list[dict[str, str]], field: str) -> int:
    return sum(1 for row in rows if row.get(field, "").strip())


def discover_roles() -> list[str]:
    roles = set(discover_workspace_roles())
    if INPUT_ROOT.exists():
        for item in INPUT_ROOT.iterdir():
            if item.is_dir():
                roles.add(normalize_role(item.name))
    return sorted(roles)


def scan_role_state(role: str) -> RoleState:
    normalized = normalize_role(role)
    input_dir = INPUT_ROOT / role
    if not input_dir.exists():
        input_dir = INPUT_ROOT / normalized
    input_count = len([p for p in input_dir.rglob("*") if p.is_file()]) if input_dir.exists() else 0
    rows = read_manifest(normalized)
    workflow_rows = read_workflow_rows(normalized, auto_sync=False)
    counts = {status: sum(1 for row in rows if row.filter_status == status) for status in ["confirmed", "review", "excluded", "deleted"]}
    reference_count = sum(1 for row in rows if row.selected_reference)
    tts_info = latest_tts_run_info(normalized)
    return RoleState(
        role=normalized,
        input_audio_count=input_count,
        raw_audio_count=len(rows),
        reference_audio_count=reference_count,
        confirmed_audio_count=counts.get("confirmed", 0),
        review_audio_count=counts.get("review", 0),
        excluded_audio_count=counts.get("excluded", 0) + counts.get("deleted", 0),
        asr_rows=count_stage_rows(workflow_rows, "asr_status"),
        emotion_rows=count_stage_rows(workflow_rows, "emotion_status"),
        has_role_excel=any(row.get("library_description", "").strip() or row.get("library_keywords", "").strip() for row in workflow_rows),
        tts_run_count=int(tts_info["tts_run_count"]),
        latest_tts_run=tts_info["latest_tts_run"],
        latest_synthesized_audio=tts_info["latest_synthesized_audio"],
    )


def get_project_state() -> dict[str, Any]:
    roles = discover_roles()
    return {"roles": [asdict(scan_role_state(role)) for role in roles]}
