from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def resolve_project_root() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if exe_dir.name.lower() == "dist":
            return exe_dir.parent
        return exe_dir
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = resolve_project_root()
GUI_SETTINGS_PATH = PROJECT_ROOT / "voice_gui_settings.json"

ASR_MODEL_OPTIONS = ("qwen3-asr-flash", "qwen3-asr-flash-2026-02-10", "qwen3-asr-flash-2025-09-08")
EMOTION_MODEL_OPTIONS = ("qwen3.5-omni-plus", "qwen3.5-omni-flash", "qwen3.5-omni-flash-2026-03-15")
RETRIEVAL_RERANK_MODEL_OPTIONS = ("qwen3-rerank",)
FINAL_RANKING_TEXT_MODEL_OPTIONS = (
    "qwen3.6-flash",
    "qwen3.6-flash",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
)
VOICE_ENROLLMENT_MODEL_OPTIONS = ("qwen-voice-enrollment",)
CLOUD_TTS_MODEL_OPTIONS = ("qwen3-tts-vc-2026-01-22",)


@dataclass
class AppConfig:
    project_root: str = str(PROJECT_ROOT)
    voice_python: str = ""
    runtime_environment: str = ""
    voxcpm2_model_path: str = str(PROJECT_ROOT / "VoxCPM2")
    dashscope_api_key: str = ""
    deepseek_api_key: str = ""
    last_backend: str = "voxcpm2_local_hifi"
    reference_window: int = 10
    reference_min_seconds: float = 15.0
    gap_ms: int = 300
    voxcpm2_diffusion_steps: int = 20
    voxcpm2_cfg_value: float = 2.0
    asr_model: str = "qwen3-asr-flash"
    emotion_model: str = "qwen3.5-omni-plus"
    retrieval_rerank_model: str = "qwen3-rerank"
    retrieval_text_model: str = "qwen3.6-flash"
    voice_enrollment_model: str = "qwen-voice-enrollment"
    tts_target_model: str = "qwen3-tts-vc-2026-01-22"
    synthesis_language: str = "zh"
    asr_source_language: str = "zh"

    @classmethod
    def load(cls) -> "AppConfig":
        if not GUI_SETTINGS_PATH.exists():
            config = cls()
            config.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
            config.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
            return config
        data = json.loads(GUI_SETTINGS_PATH.read_text(encoding="utf-8-sig"))
        defaults = asdict(cls())
        defaults.update({key: value for key, value in data.items() if key in defaults})
        config = cls(**defaults)
        config.project_root = str(PROJECT_ROOT)
        default_model_path = PROJECT_ROOT / "VoxCPM2"
        configured_model_path = Path(config.voxcpm2_model_path).expanduser()
        if not str(configured_model_path).strip() or not configured_model_path.exists():
            config.voxcpm2_model_path = str(default_model_path)
        if not config.dashscope_api_key:
            config.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not config.deepseek_api_key:
            config.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        return config

    def save(self) -> None:
        GUI_SETTINGS_PATH.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.dashscope_api_key.strip():
            env["DASHSCOPE_API_KEY"] = self.dashscope_api_key.strip()
        if self.deepseek_api_key.strip():
            env["DEEPSEEK_API_KEY"] = self.deepseek_api_key.strip()
        env["PYTHONIOENCODING"] = "utf-8"
        return env


def path_from_config(value: str) -> Path:
    return Path(value).expanduser().resolve()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump())
        except Exception:
            return str(value)
    if hasattr(value, "to_dict"):
        try:
            return to_jsonable(value.to_dict())
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return str(value)
