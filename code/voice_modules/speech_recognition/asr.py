from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path
from typing import Any

import dashscope
from dashscope import MultiModalConversation

from voice_modules.common.audio_manifest import resolve_audio_path_for_row
from voice_modules.common.workflow_db import (
    ASR_CHINESE_TEXT,
    ASR_ERROR,
    ASR_FILE,
    ASR_LANGUAGE,
    ASR_ORIGINAL_TEXT,
    TRANSLATION_ERROR,
    load_asr_rows as load_asr_view_rows,
    read_workflow_rows,
    save_asr_view_rows,
)


ASR_FIELDS = [ASR_FILE, ASR_LANGUAGE, ASR_ORIGINAL_TEXT, ASR_CHINESE_TEXT, ASR_ERROR, TRANSLATION_ERROR]
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def ensure_api_key() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY。")
    dashscope.api_key = api_key


def infer_mime_type(audio_path: Path) -> str:
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }.get(audio_path.suffix.lower(), "application/octet-stream")


def audio_to_data_uri(audio_path: Path) -> str:
    return f"data:{infer_mime_type(audio_path)};base64,{base64.b64encode(audio_path.read_bytes()).decode('utf-8')}"


def response_error(response: Any) -> str:
    status_code = getattr(response, "status_code", None)
    if status_code is None and isinstance(response, dict):
        status_code = response.get("status_code")
    if status_code == 200:
        return ""
    message = getattr(response, "message", None)
    if message is None and isinstance(response, dict):
        message = response.get("message", "")
    request_id = getattr(response, "request_id", None)
    if request_id is None and isinstance(response, dict):
        request_id = response.get("request_id", "")
    return f"status_code={status_code}, message={message}, request_id={request_id}"


def extract_text(response: Any) -> str:
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    choices = getattr(output, "choices", None) if output else None
    if choices is None and isinstance(output, dict):
        choices = output.get("choices", [])
    if not choices:
        return ""
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if not content:
        return ""
    texts = []
    for item in content:
        if isinstance(item, dict) and item.get("text"):
            texts.append(str(item["text"]).strip())
    return "\n".join(text for text in texts if text)


def parse_status_code(error_text: str) -> int | None:
    match = re.search(r"status_code=(\d+)", error_text or "")
    return int(match.group(1)) if match else None


def response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def add_usage(total: dict[str, int], incoming: dict[str, int]) -> None:
    for key in ["input_tokens", "output_tokens", "total_tokens"]:
        total[key] += int(incoming.get(key, 0) or 0)


def recognize_once(audio_path: Path, model: str, language: str) -> tuple[str, str, dict[str, int]]:
    response = MultiModalConversation.call(
        model=model,
        messages=[{"role": "user", "content": [{"audio": audio_to_data_uri(audio_path)}]}],
        result_format="message",
        asr_options={"language": language},
    )
    error = response_error(response)
    if error:
        return "", error, response_usage(response)
    return extract_text(response), "", response_usage(response)


def recognize_with_retry(audio_path: Path, model: str, language: str, max_attempts: int = 4, base_sleep: float = 2.0) -> tuple[str, str, dict[str, int]]:
    last_error = ""
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for attempt in range(1, max_attempts + 1):
        text, error, usage = recognize_once(audio_path, model, language)
        add_usage(total_usage, usage)
        if not error:
            return text, "", total_usage
        last_error = error
        if attempt >= max_attempts or parse_status_code(error) not in RETRYABLE_STATUS_CODES:
            break
        time.sleep(base_sleep * (2 ** (attempt - 1)))
    return "", last_error, total_usage


def load_asr_rows(role: str) -> list[dict[str, str]]:
    return load_asr_view_rows(role)


def save_asr_rows(role: str, rows: list[dict[str, Any]]) -> Path:
    return save_asr_view_rows(role, rows)


def run_asr(
    role: str,
    model: str = "qwen3-asr-flash",
    language: str = "zh",
    limit: int | None = None,
    sha256: str = "",
    force: bool = False,
) -> list[dict[str, str]]:
    ensure_api_key()
    target_sha256 = sha256.strip()
    workflow_rows = [
        row
        for row in read_workflow_rows(role)
        if row.get("filter_status") == "confirmed"
        and resolve_audio_path_for_row(
            role=row.get("role", "") or role,
            audio_path=row.get("audio_path", ""),
            stored_name=row.get("stored_name", ""),
            file_name=row.get("file_name", ""),
        ).exists()
    ]
    if target_sha256 and not any(row.get("sha256") == target_sha256 for row in workflow_rows):
        raise ValueError("所选音频不存在、未通过筛选或音频文件缺失，无法运行单条 ASR。")

    saved_rows = load_asr_rows(role)
    saved_by_hash = {row.get("sha256", ""): row for row in saved_rows if row.get("sha256")}
    preview_rows: list[dict[str, str]] = []
    pending_rows: list[dict[str, str]] = []

    for row in workflow_rows:
        preview_row = {
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
        existing = saved_by_hash.get(preview_row["sha256"])
        if existing is not None:
            preview_row.update(existing)
        preview_rows.append(preview_row)
        should_run_row = not target_sha256 or row.get("sha256") == target_sha256
        if should_run_row and (force or not row.get("asr_original_text", "").strip() or row.get("asr_language") != language):
            pending_rows.append(preview_row)

    if limit is not None:
        pending_rows = pending_rows[:limit]
    total = len(pending_rows)
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    scope_text = "选中音频" if target_sha256 else "confirmed 音频"
    print(f"[ASR] 开始运行，共 {total} 条 {scope_text}。", flush=True)
    print(f"[ASR] 原始语言={language}", flush=True)

    for index, preview_row in enumerate(pending_rows, start=1):
        audio_path = Path(preview_row["file_path"])
        transcript, error, usage = recognize_with_retry(audio_path, model, language)
        add_usage(usage_totals, usage)
        preview_row["model"] = model
        preview_row[ASR_LANGUAGE] = language
        preview_row[ASR_ORIGINAL_TEXT] = transcript
        preview_row[ASR_ERROR] = error
        preview_row["translation_model"] = "direct_copy" if language == "zh" and transcript else ""
        if language == "zh":
            preview_row[ASR_CHINESE_TEXT] = transcript
            preview_row[TRANSLATION_ERROR] = ""
        else:
            preview_row[ASR_CHINESE_TEXT] = ""
            preview_row[TRANSLATION_ERROR] = ""
        if index % 5 == 0 or index == total:
            print(f"[ASR] 已处理 {index}/{total} 条。", flush=True)

    error_count = sum(1 for row in pending_rows if row.get(ASR_ERROR))
    print(f"[ASR] 运行完成。成功 {len(pending_rows) - error_count} 条，失败 {error_count} 条。", flush=True)
    if error_count:
        failed_items = [row.get(ASR_FILE, "") for row in pending_rows if row.get(ASR_ERROR)]
        print(f"[ASR] 失败条目: {', '.join(name for name in failed_items if name)}", flush=True)
    if usage_totals["total_tokens"] > 0:
        print(
            f"[ASR] API token: model={model} input={usage_totals['input_tokens']} "
            f"output={usage_totals['output_tokens']} total={usage_totals['total_tokens']}",
            flush=True,
        )
    if not target_sha256:
        save_asr_rows(role, preview_rows)
        print("[ASR] 全量运行结果已自动保存。", flush=True)
    else:
        print("[ASR] 本次结果仅供预览，点击“全部保存”后才会正式写入 workflow_db.csv。", flush=True)
    return preview_rows
