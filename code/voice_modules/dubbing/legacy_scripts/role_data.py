from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROLE_CSV_SUFFIX = "_情感标定.csv"
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SORTED_AUDIO_ROOT = WORKSPACE_ROOT / "4.整理音频"
ROLE_DATA_ROOT = PROJECT_ROOT / "角色数据"

REQUIRED_ROLE_CSV_COLUMNS = {
    "索引",
    "语音文件",
    "语音文本",
    "音频路径",
    "自然语言描述",
    "情绪语气",
    "音频表达技巧",
}

ROLE_ALIASES = {
    "莫宁": "莫宁",
    "moning": "莫宁",
    "morning": "莫宁",
    "m": "莫宁",
    "1": "莫宁",
    "尤诺": "尤诺",
    "younuo": "尤诺",
    "yuno": "尤诺",
    "y": "尤诺",
    "2": "尤诺",
}


@dataclass(frozen=True)
class RolePaths:
    role: str
    audio_dir: Path
    csv_path: Path
    excel_path: Path
    output_dir: Path


def discover_available_roles() -> list[str]:
    roles: set[str] = set()
    if SORTED_AUDIO_ROOT.exists():
        for csv_path in SORTED_AUDIO_ROOT.glob(f"*{ROLE_CSV_SUFFIX}"):
            role = csv_path.name[: -len(ROLE_CSV_SUFFIX)]
            if role:
                roles.add(role)
        for audio_dir in SORTED_AUDIO_ROOT.iterdir():
            if audio_dir.is_dir() and not audio_dir.name.startswith("__"):
                roles.add(audio_dir.name)
    return sorted(roles)


def resolve_role_paths(role: str) -> RolePaths:
    normalized_role = role.strip()
    if not normalized_role:
        raise ValueError("角色名不能为空。")
    normalized_role = ROLE_ALIASES.get(normalized_role.lower(), normalized_role)

    csv_path = SORTED_AUDIO_ROOT / f"{normalized_role}{ROLE_CSV_SUFFIX}"
    audio_dir = SORTED_AUDIO_ROOT / normalized_role
    excel_path = ROLE_DATA_ROOT / normalized_role / "情绪关键词.xlsx"
    output_dir = PROJECT_ROOT / "output" / normalized_role

    if not csv_path.exists():
        raise FileNotFoundError(f"角色标定文件不存在: role={normalized_role} path={csv_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"角色音频目录不存在: role={normalized_role} path={audio_dir}")

    return RolePaths(
        role=normalized_role,
        audio_dir=audio_dir.resolve(),
        csv_path=csv_path.resolve(),
        excel_path=excel_path.resolve(),
        output_dir=output_dir.resolve(),
    )


def build_keyword_text(row: pd.Series) -> str:
    parts: list[str] = []
    for column in ("情绪语气", "音频表达技巧"):
        value = str(row.get(column, "") or "").strip()
        if value:
            parts.append(value)
    return "，".join(parts)


def prepare_role_excel(role: str, *, force: bool = False) -> RolePaths:
    role_paths = resolve_role_paths(role)
    role_paths.excel_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        not force
        and role_paths.excel_path.exists()
        and role_paths.excel_path.stat().st_mtime >= role_paths.csv_path.stat().st_mtime
    ):
        return role_paths

    df = pd.read_csv(role_paths.csv_path, encoding="utf-8-sig")
    missing = REQUIRED_ROLE_CSV_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"角色标定文件缺少字段: role={role_paths.role} path={role_paths.csv_path} missing={sorted(missing)}"
        )

    output_df = pd.DataFrame(
        {
            "索引": df["索引"],
            "语音文件": df["语音文件"],
            "语音文本": df["语音文本"],
            "音频路径": df["音频路径"],
            "语音描述-音频理解模型": df["自然语言描述"],
            "关键词": df.apply(build_keyword_text, axis=1),
        }
    )
    output_df.to_excel(role_paths.excel_path, index=False)
    return role_paths


def ensure_role_excel(role: str) -> RolePaths:
    return prepare_role_excel(role, force=False)


def role_summary_dict(role_paths: RolePaths) -> dict[str, Any]:
    return {
        "role": role_paths.role,
        "role_audio_dir": str(role_paths.audio_dir),
        "role_csv_path": str(role_paths.csv_path),
        "role_excel_path": str(role_paths.excel_path),
        "role_output_dir": str(role_paths.output_dir),
    }
