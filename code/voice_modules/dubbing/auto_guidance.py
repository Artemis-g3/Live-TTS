from __future__ import annotations

import json
import os
import time
from pathlib import Path

from openai import OpenAI

from voice_modules.common.audio_manifest import PROJECT_ROOT


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PROMPT_PATH = PROJECT_ROOT / "code" / "voice_modules" / "dubbing" / "prompts" / "auto_guidance_prompt.txt"
RETRY_MAX_ATTEMPTS = int(os.getenv("AUTO_GUIDANCE_RETRY_MAX", "3"))
RETRY_BASE_SLEEP = float(os.getenv("AUTO_GUIDANCE_RETRY_SLEEP", "1.5"))


def _make_text_client(text_model: str) -> OpenAI:
    if text_model.strip().lower().startswith("deepseek"):
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY。请先在设置页填写 Deepseek API Key。")
        return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY。")
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)


def _load_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"未找到自动指导提示词文件: {PROMPT_PATH}")
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"自动指导提示词文件为空: {PROMPT_PATH}")
    return prompt


def _parse_guidance_output(output_text: str) -> list[str]:
    text = output_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end + 1])
        else:
            raise ValueError(f"无法解析模型返回的 JSON: {output_text[:200]}")
    keywords = data.get("guidance_keywords", [])
    if isinstance(keywords, str):
        keywords = [keywords]
    if not isinstance(keywords, list):
        raise ValueError(f"guidance_keywords 应为数组，实际为: {type(keywords)}")
    return [str(k).strip() for k in keywords if str(k).strip()]


def generate_auto_guidance(transcript: str, text_model: str) -> str:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("配音台词不能为空。")
    prompt_template = _load_prompt()
    input_text = prompt_template.replace("{{transcript}}", transcript)
    client = _make_text_client(text_model)
    last_error: Exception | None = None
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=text_model,
                messages=[{"role": "user", "content": input_text}],
            )
            output_text = (response.choices[0].message.content or "").strip()
            if not output_text:
                raise RuntimeError("模型返回为空。")
            keywords = _parse_guidance_output(output_text)
            return "，".join(keywords)
        except Exception as exc:
            last_error = exc
            if attempt >= RETRY_MAX_ATTEMPTS:
                break
            sleep_seconds = RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            print(f"[AutoGuidance] 第 {attempt}/{RETRY_MAX_ATTEMPTS} 次失败: {exc}", flush=True)
            print(f"[AutoGuidance] {sleep_seconds:.1f}s 后重试。", flush=True)
            time.sleep(sleep_seconds)
    assert last_error is not None
    raise last_error
