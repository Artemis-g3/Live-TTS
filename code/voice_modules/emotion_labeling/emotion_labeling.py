from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from voice_modules.common.audio_manifest import PROJECT_ROOT, metadata_dir, normalize_role, resolve_audio_path_for_row
from voice_modules.common.workflow_db import (
    AUDIO_FILE,
    AUDIO_PATH,
    DELIVERY,
    DESCRIPTION,
    IDX,
    KEYWORDS,
    TONE,
    TRANSCRIPT,
    load_emotion_rows as load_emotion_view_rows,
)
from voice_modules.common.workflow_db import read_workflow_rows, save_emotion_view_rows


MODEL = "qwen3.5-omni-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
PROMPT_PATH = PROJECT_ROOT / "code" / "voice_modules" / "emotion_labeling" / "prompts" / "qwen35_omni_audio_prompt.txt"
DESCRIPTION_PROMPT_PATH = PROJECT_ROOT / "code" / "voice_modules" / "emotion_labeling" / "prompts" / "qwen35_omni_audio_prompt_description.txt"
KEYWORD_PROMPT_PATH = PROJECT_ROOT / "code" / "voice_modules" / "emotion_labeling" / "prompts" / "qwen35_omni_audio_prompt_keyword.txt"
EMOTION_FIELDS = [IDX, AUDIO_FILE, TRANSCRIPT, AUDIO_PATH, DESCRIPTION, TONE, DELIVERY, KEYWORDS]
RETRY_MAX_ATTEMPTS = int(os.getenv("EMO_RETRY_MAX_ATTEMPTS", "5"))
RETRY_BASE_SLEEP = float(os.getenv("EMO_RETRY_BASE_SLEEP", "2"))


def make_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY。")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def make_text_client(text_model: str) -> OpenAI:
    if text_model.strip().lower().startswith("deepseek"):
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY。请先在设置页填写 Deepseek API Key。")
        return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY。")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def load_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"未找到情感标定提示词文件: {PROMPT_PATH}")
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"情感标定提示词文件为空: {PROMPT_PATH}")
    return prompt


def load_description_prompt() -> str:
    if not DESCRIPTION_PROMPT_PATH.exists():
        raise FileNotFoundError(f"未找到描述生成提示词文件: {DESCRIPTION_PROMPT_PATH}")
    prompt = DESCRIPTION_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"描述生成提示词文件为空: {DESCRIPTION_PROMPT_PATH}")
    return prompt


def load_keyword_prompt() -> str:
    if not KEYWORD_PROMPT_PATH.exists():
        raise FileNotFoundError(f"未找到关键词提取提示词文件: {KEYWORD_PROMPT_PATH}")
    prompt = KEYWORD_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"关键词提取提示词文件为空: {KEYWORD_PROMPT_PATH}")
    return prompt


def encode_audio_as_data_url(audio_path: Path) -> tuple[str, str]:
    fmt = audio_path.suffix.lower().lstrip(".")
    mime_type, _ = mimetypes.guess_type(audio_path.name)
    if not mime_type:
        mime_type = f"audio/{fmt}"
    return fmt, f"data:{mime_type};base64,{base64.b64encode(audio_path.read_bytes()).decode('utf-8')}"


def extract_text_delta(delta: Any) -> str:
    if delta is None:
        return ""
    content = getattr(delta, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
            elif isinstance(item, dict) and item.get("content"):
                parts.append(item["content"])
            elif hasattr(item, "text") and item.text:
                parts.append(item.text)
        return "".join(parts)
    return ""


def extract_json_text(output_text: str) -> str:
    text = output_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"模型输出不含 JSON: {output_text}")
    return text[start : end + 1]


def parse_model_output(output_text: str) -> dict[str, Any]:
    payload = json.loads(extract_json_text(output_text))
    if not isinstance(payload, dict):
        raise ValueError("模型输出不是 JSON object")
    keywords = payload.get("keywords") or {}
    tone = keywords.get("tone_emotion_attitude") or []
    prosody = keywords.get("prosody_delivery") or []
    if isinstance(tone, str):
        tone = [tone]
    if isinstance(prosody, str):
        prosody = [prosody]
    if not isinstance(tone, list) or not isinstance(prosody, list):
        raise ValueError("模型输出 keywords 字段格式不正确")
    return {
        "voice_description": str(payload.get("voice_description") or "").strip(),
        "tone_emotion_attitude": "，".join(str(item).strip() for item in tone if str(item).strip()),
        "prosody_delivery": "，".join(str(item).strip() for item in prosody if str(item).strip()),
        "parsed_json": payload,
    }


def parse_description_output(output_text: str) -> dict[str, Any]:
    payload = json.loads(extract_json_text(output_text))
    if not isinstance(payload, dict):
        raise ValueError("描述模型输出不是 JSON object")
    return {
        "voice_description": str(payload.get("voice_description") or "").strip(),
        "parsed_json": payload,
    }


def parse_keyword_output(output_text: str) -> dict[str, Any]:
    payload = json.loads(extract_json_text(output_text))
    if not isinstance(payload, dict):
        raise ValueError("关键词模型输出不是 JSON object")
    keywords = payload.get("keywords") or {}
    tone = keywords.get("tone_emotion_attitude") or []
    prosody = keywords.get("prosody_delivery") or []
    if isinstance(tone, str):
        tone = [tone]
    if isinstance(prosody, str):
        prosody = [prosody]
    if not isinstance(tone, list) or not isinstance(prosody, list):
        raise ValueError("关键词模型输出 keywords 字段格式不正确")
    return {
        "tone_emotion_attitude": "，".join(str(item).strip() for item in tone if str(item).strip()),
        "prosody_delivery": "，".join(str(item).strip() for item in prosody if str(item).strip()),
        "parsed_json": payload,
    }


def render_prompt(prompt: str, transcript: str) -> str:
    if "{{语音文本}}" in prompt:
        return prompt.replace("{{语音文本}}", transcript)
    return f"{prompt}\n\n文本如下：\n{transcript}"


def render_keyword_prompt(prompt: str, voice_description: str) -> str:
    if "{{voice_description}}" in prompt:
        return prompt.replace("{{voice_description}}", voice_description)
    return f"{prompt}\n\nvoice_description:\n{voice_description}"


def usage_to_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def add_usage(total: dict[str, int], incoming: dict[str, int]) -> None:
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        total[key] += int(incoming.get(key, 0) or 0)


def call_model_once(client: OpenAI, prompt: str, audio_path: Path, transcript: str, model: str) -> dict[str, Any]:
    audio_format, data_url = encode_audio_as_data_url(audio_path)
    input_text = render_prompt(prompt, transcript)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": data_url, "format": audio_format}},
                    {"type": "text", "text": input_text},
                ],
            }
        ],
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
    )
    parts: list[str] = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for chunk in stream:
        if not chunk.choices:
            add_usage(usage_totals, usage_to_dict(getattr(chunk, "usage", None)))
            continue
        text = extract_text_delta(chunk.choices[0].delta)
        if text:
            parts.append(text)
        add_usage(usage_totals, usage_to_dict(getattr(chunk, "usage", None)))
    output_text = "".join(parts).strip()
    parsed = parse_model_output(output_text)
    return {"input_text": input_text, "output_text": output_text, "usage": usage_totals, **parsed}


def call_model(client: OpenAI, prompt: str, audio_path: Path, transcript: str, model: str = MODEL) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return call_model_once(client, prompt, audio_path, transcript, model)
        except Exception as exc:
            last_error = exc
            if attempt >= RETRY_MAX_ATTEMPTS:
                break
            sleep_seconds = RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            print(f"[Emotion] {audio_path.name} 第 {attempt}/{RETRY_MAX_ATTEMPTS} 次失败: {exc}", flush=True)
            print(f"[Emotion] {sleep_seconds:.1f}s 后重试。", flush=True)
            time.sleep(sleep_seconds)
    assert last_error is not None
    raise last_error


def call_keyword_model_once(client: OpenAI, prompt: str, voice_description: str, text_model: str) -> dict[str, Any]:
    input_text = render_keyword_prompt(prompt, voice_description)
    response = client.chat.completions.create(
        model=text_model,
        messages=[{"role": "user", "content": input_text}],
    )
    output_text = (response.choices[0].message.content or "").strip()
    if not output_text:
        raise RuntimeError("关键词提取模型返回为空。")
    usage = usage_to_dict(getattr(response, "usage", None))
    parsed = parse_keyword_output(output_text)
    return {"input_text": input_text, "output_text": output_text, "usage": usage, **parsed}


def call_keyword_model(client: OpenAI, prompt: str, voice_description: str, text_model: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return call_keyword_model_once(client, prompt, voice_description, text_model)
        except Exception as exc:
            last_error = exc
            if attempt >= RETRY_MAX_ATTEMPTS:
                break
            sleep_seconds = RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            print(f"[Emotion KW] 第 {attempt}/{RETRY_MAX_ATTEMPTS} 次失败: {exc}", flush=True)
            print(f"[Emotion KW] {sleep_seconds:.1f}s 后重试。", flush=True)
            time.sleep(sleep_seconds)
    assert last_error is not None
    raise last_error


def load_emotion_rows(role: str) -> list[dict[str, str]]:
    return load_emotion_view_rows(role)


def save_emotion_rows(role: str, rows: list[dict[str, Any]]) -> Path:
    model = ""
    for row in rows:
        if str(row.get("model", "") or "").strip():
            model = str(row.get("model", "") or "").strip()
            break
    return save_emotion_view_rows(role, rows, model=model)


def run_record_dir(role: str) -> Path:
    return metadata_dir(role) / "emotion_runs"


def run_emotion_labeling(
    role: str,
    limit: int | None = None,
    model: str = MODEL,
    text_model: str = "",
    sha256: str = "",
    force: bool = False,
) -> list[dict[str, Any]]:
    normalized_role = normalize_role(role)
    target_sha256 = sha256.strip()
    desc_prompt = load_description_prompt()
    kw_prompt = load_keyword_prompt()
    multimodal_client = make_client()
    text_client = make_text_client(text_model) if text_model.strip() else None
    if text_client is None:
        print("[Emotion] 未配置文本模型，将跳过关键词提取。", flush=True)
    workflow_rows = [
        row
        for row in read_workflow_rows(normalized_role)
        if row.get("filter_status") == "confirmed" and row.get("asr_status") == "done" and row.get("asr_transcript", "").strip()
    ]
    if target_sha256:
        if not any(row.get("sha256") == target_sha256 for row in workflow_rows):
            raise ValueError("所选音频尚未完成 ASR、未通过筛选或不存在，无法运行单条情感标定。")
    preview_rows = load_emotion_rows(normalized_role)
    preview_by_hash = {row.get("sha256", ""): row for row in preview_rows if row.get("sha256")}
    rows_for_display: list[dict[str, Any]] = []
    pending_rows: list[dict[str, str]] = []
    for index, workflow_row in enumerate(sorted(workflow_rows, key=lambda row: (row.get("file_name", ""), row.get("sha256", ""))), start=1):
        view_row = {
            "sha256": workflow_row.get("sha256", ""),
            "索引": str(index),
            "语音文件": workflow_row.get("file_name", ""),
            "语音文本": workflow_row.get("asr_transcript", ""),
            "音频路径": str(
                resolve_audio_path_for_row(
                    role=workflow_row.get("role", "") or normalized_role,
                    audio_path=workflow_row.get("audio_path", ""),
                    stored_name=workflow_row.get("stored_name", ""),
                    file_name=workflow_row.get("file_name", ""),
                )
            ),
            "自然语言描述": workflow_row.get("emotion_description", ""),
            "情绪语气": workflow_row.get("emotion_tone", ""),
            "音频表达技巧": workflow_row.get("emotion_delivery", ""),
            "model": workflow_row.get("emotion_model", ""),
            "error": workflow_row.get("emotion_error", ""),
        }
        existing = preview_by_hash.get(view_row["sha256"])
        if existing is not None:
            view_row.update(existing)
        rows_for_display.append(view_row)
        should_run_row = not target_sha256 or workflow_row.get("sha256") == target_sha256
        if should_run_row and (force or workflow_row.get("emotion_status") != "done"):
            pending_rows.append(view_row)
    if limit is not None:
        pending_rows = pending_rows[:limit]
    total = len(rows_for_display)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    records_dir = run_record_dir(normalized_role)
    records_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = records_dir / f"emotion_run_{run_tag}.jsonl"
    summary_path = records_dir / f"emotion_summary_{run_tag}.json"

    scope_text = "选中音频" if target_sha256 else "ASR 结果"
    run_total = len(pending_rows) if target_sha256 else total
    print(f"[Emotion] 开始运行，共 {run_total} 条 {scope_text}。", flush=True)
    print(f"[Emotion] 已完成 {total - len(pending_rows)} 条，待处理 {len(pending_rows)} 条。", flush=True)
    if text_client:
        print(f"[Emotion] 多模态模型={model}, 关键词模型={text_model}", flush=True)
    else:
        print(f"[Emotion] 多模态模型={model}", flush=True)
    processed_this_run = 0
    step1_error_count = 0
    step2_error_count = 0
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    kw_usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with jsonl_path.open("w", encoding="utf-8") as log_file:
        for index, row in enumerate(pending_rows, start=1):
            audio_path = Path(row[AUDIO_PATH])
            if not audio_path.exists():
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")

            # Step 1: 多模态模型生成自然语言描述
            desc_result = None
            voice_description = ""
            try:
                desc_result = call_model(multimodal_client, desc_prompt, audio_path, row[TRANSCRIPT], model)
                voice_description = desc_result["voice_description"]
                row[DESCRIPTION] = voice_description
                row["model"] = model
                add_usage(usage_totals, desc_result.get("usage", {}))
            except Exception as exc:
                desc_result = None
                row[DESCRIPTION] = ""
                row[TONE] = ""
                row[DELIVERY] = ""
                row[KEYWORDS] = ""
                row["model"] = model
                row["error"] = f"Step1-描述生成: {exc}"
                step1_error_count += 1
                log_file.write(
                    json.dumps(
                        {
                            "role": normalized_role,
                            "audio_file": row[AUDIO_FILE],
                            "audio_path": str(audio_path),
                            "transcript": row[TRANSCRIPT],
                            "multimodal_model": model,
                            "text_model": text_model if text_client else "",
                            "desc_input": "",
                            "desc_output": "",
                            "desc_parsed_json": None,
                            "desc_usage": {},
                            "kw_input": "",
                            "kw_output": "",
                            "kw_parsed_json": None,
                            "kw_usage": {},
                            "error": row["error"],
                            "preview_row": {
                                "sha256": row["sha256"],
                                DESCRIPTION: row[DESCRIPTION],
                                KEYWORDS: row[KEYWORDS],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log_file.flush()
                processed_this_run += 1
                if index % 5 == 0 or index == len(pending_rows):
                    print(f"[Emotion] 已处理 {index}/{len(pending_rows)} 条待处理样本。", flush=True)
                time.sleep(0.1)
                continue

            # Step 2: 文本模型提取关键词
            kw_result = None
            if text_client and voice_description:
                try:
                    kw_result = call_keyword_model(text_client, kw_prompt, voice_description, text_model)
                    row[TONE] = kw_result["tone_emotion_attitude"]
                    row[DELIVERY] = kw_result["prosody_delivery"]
                    row[KEYWORDS] = "，".join(part for part in [row[TONE], row[DELIVERY]] if part)
                    row["error"] = ""
                    add_usage(kw_usage_totals, kw_result.get("usage", {}))
                except Exception as exc:
                    kw_result = None
                    row[TONE] = ""
                    row[DELIVERY] = ""
                    row[KEYWORDS] = ""
                    row["error"] = f"Step2-关键词提取: {exc}"
                    step2_error_count += 1
            else:
                row[TONE] = ""
                row[DELIVERY] = ""
                row[KEYWORDS] = ""
                if not text_client:
                    row["error"] = ""

            log_file.write(
                json.dumps(
                    {
                        "role": normalized_role,
                        "audio_file": row[AUDIO_FILE],
                        "audio_path": str(audio_path),
                        "transcript": row[TRANSCRIPT],
                        "multimodal_model": model,
                        "text_model": text_model if text_client else "",
                        "desc_input": desc_result["input_text"] if desc_result is not None else "",
                        "desc_output": desc_result["output_text"] if desc_result is not None else "",
                        "desc_parsed_json": desc_result["parsed_json"] if desc_result is not None else None,
                        "desc_usage": desc_result.get("usage", {}) if desc_result is not None else {},
                        "kw_input": kw_result["input_text"] if kw_result is not None else "",
                        "kw_output": kw_result["output_text"] if kw_result is not None else "",
                        "kw_parsed_json": kw_result["parsed_json"] if kw_result is not None else None,
                        "kw_usage": kw_result.get("usage", {}) if kw_result is not None else {},
                        "error": row["error"],
                        "preview_row": {
                            "sha256": row["sha256"],
                            DESCRIPTION: row[DESCRIPTION],
                            KEYWORDS: row[KEYWORDS],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log_file.flush()
            processed_this_run += 1
            if index % 5 == 0 or index == len(pending_rows):
                print(f"[Emotion] 已处理 {index}/{len(pending_rows)} 条待处理样本。", flush=True)
            time.sleep(0.1)

    all_rows = load_emotion_rows(normalized_role)
    preview_map = {row.get("sha256", ""): row for row in rows_for_display if row.get("sha256")}
    for row in pending_rows:
        preview_map[row["sha256"]] = row
    preview_rows = [preview_map[row.get("sha256", "")] for row in all_rows if row.get("sha256", "") in preview_map]
    total_error = step1_error_count + step2_error_count
    summary = {
        "role": normalized_role,
        "multimodal_model": model,
        "text_model": text_model if text_client else "",
        "desc_prompt_path": str(DESCRIPTION_PROMPT_PATH),
        "kw_prompt_path": str(KEYWORD_PROMPT_PATH),
        "record_path": str(jsonl_path),
        "total_asr_rows": total,
        "completed_before_run": total - len(pending_rows),
        "processed_this_run": processed_this_run,
        "step1_error_count": step1_error_count,
        "step2_error_count": step2_error_count,
        "preview_rows": len(preview_rows),
        "error_rows": total_error,
        "desc_token_usage": usage_totals,
        "kw_token_usage": kw_usage_totals,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Emotion] 运行完成。本次处理 {processed_this_run} 条。", flush=True)
    if usage_totals["total_tokens"] > 0:
        print(
            f"[Emotion] Step1 多模态 API token: model={model} prompt={usage_totals['prompt_tokens']} "
            f"completion={usage_totals['completion_tokens']} total={usage_totals['total_tokens']}",
            flush=True,
        )
    if kw_usage_totals["total_tokens"] > 0:
        print(
            f"[Emotion] Step2 关键词 API token: model={text_model} prompt={kw_usage_totals['prompt_tokens']} "
            f"completion={kw_usage_totals['completion_tokens']} total={kw_usage_totals['total_tokens']}",
            flush=True,
        )
    if step1_error_count or step2_error_count:
        print(f"[Emotion] 错误: Step1={step1_error_count}, Step2={step2_error_count}", flush=True)
    print(f"[Emotion] 运行记录: {jsonl_path}", flush=True)
    print("[Emotion] 本次结果仅供预览，点击\"保存情感标定\"后才会正式写入 workflow_db.csv。", flush=True)
    return preview_rows


def run_description_only(
    role: str,
    limit: int | None = None,
    model: str = MODEL,
    sha256: str = "",
    force: bool = False,
) -> list[dict[str, Any]]:
    normalized_role = normalize_role(role)
    target_sha256 = sha256.strip()
    desc_prompt = load_description_prompt()
    multimodal_client = make_client()
    workflow_rows = [
        row
        for row in read_workflow_rows(normalized_role)
        if row.get("filter_status") == "confirmed" and row.get("asr_status") == "done" and row.get("asr_transcript", "").strip()
    ]
    if target_sha256:
        if not any(row.get("sha256") == target_sha256 for row in workflow_rows):
            raise ValueError("所选音频尚未完成 ASR、未通过筛选或不存在，无法运行情感描述。")
    preview_rows = load_emotion_rows(normalized_role)
    preview_by_hash = {row.get("sha256", ""): row for row in preview_rows if row.get("sha256")}
    rows_for_display: list[dict[str, Any]] = []
    pending_rows: list[dict[str, str]] = []
    for index, workflow_row in enumerate(sorted(workflow_rows, key=lambda row: (row.get("file_name", ""), row.get("sha256", ""))), start=1):
        view_row = {
            "sha256": workflow_row.get("sha256", ""),
            "索引": str(index),
            "语音文件": workflow_row.get("file_name", ""),
            "语音文本": workflow_row.get("asr_transcript", ""),
            "音频路径": str(
                resolve_audio_path_for_row(
                    role=workflow_row.get("role", "") or normalized_role,
                    audio_path=workflow_row.get("audio_path", ""),
                    stored_name=workflow_row.get("stored_name", ""),
                    file_name=workflow_row.get("file_name", ""),
                )
            ),
            "自然语言描述": workflow_row.get("emotion_description", ""),
            "情绪语气": workflow_row.get("emotion_tone", ""),
            "音频表达技巧": workflow_row.get("emotion_delivery", ""),
            "model": workflow_row.get("emotion_model", ""),
            "error": workflow_row.get("emotion_error", ""),
        }
        existing = preview_by_hash.get(view_row["sha256"])
        if existing is not None:
            view_row.update(existing)
        rows_for_display.append(view_row)
        should_run_row = not target_sha256 or workflow_row.get("sha256") == target_sha256
        if should_run_row and (force or not workflow_row.get("emotion_description", "").strip()):
            pending_rows.append(view_row)
    if limit is not None:
        pending_rows = pending_rows[:limit]
    total = len(rows_for_display)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    records_dir = run_record_dir(normalized_role)
    records_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = records_dir / f"emotion_desc_{run_tag}.jsonl"
    summary_path = records_dir / f"emotion_desc_summary_{run_tag}.json"

    scope_text = "选中音频" if target_sha256 else "ASR 结果"
    run_total = len(pending_rows) if target_sha256 else total
    print(f"[Emotion Desc] 开始运行，共 {run_total} 条 {scope_text}。", flush=True)
    print(f"[Emotion Desc] 已完成 {total - len(pending_rows)} 条，待处理 {len(pending_rows)} 条。", flush=True)
    print(f"[Emotion Desc] 多模态模型={model}", flush=True)
    processed_this_run = 0
    error_count = 0
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with jsonl_path.open("w", encoding="utf-8") as log_file:
        for index, row in enumerate(pending_rows, start=1):
            audio_path = Path(row[AUDIO_PATH])
            if not audio_path.exists():
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            desc_result = None
            try:
                desc_result = call_model(multimodal_client, desc_prompt, audio_path, row[TRANSCRIPT], model)
                row[DESCRIPTION] = desc_result["voice_description"]
                row["model"] = model
                row["error"] = ""
                add_usage(usage_totals, desc_result.get("usage", {}))
            except Exception as exc:
                desc_result = None
                row[DESCRIPTION] = ""
                row["error"] = f"描述: {exc}"
                error_count += 1

            log_file.write(
                json.dumps(
                    {
                        "role": normalized_role,
                        "audio_file": row[AUDIO_FILE],
                        "audio_path": str(audio_path),
                        "transcript": row[TRANSCRIPT],
                        "model": model,
                        "desc_input": desc_result["input_text"] if desc_result is not None else "",
                        "desc_output": desc_result["output_text"] if desc_result is not None else "",
                        "desc_parsed_json": desc_result["parsed_json"] if desc_result is not None else None,
                        "desc_usage": desc_result.get("usage", {}) if desc_result is not None else {},
                        "error": row["error"],
                        "preview_row": {
                            "sha256": row["sha256"],
                            DESCRIPTION: row[DESCRIPTION],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log_file.flush()
            processed_this_run += 1
            if index % 5 == 0 or index == len(pending_rows):
                print(f"[Emotion Desc] 已处理 {index}/{len(pending_rows)} 条。", flush=True)
            time.sleep(0.1)

    preview_map = {row.get("sha256", ""): row for row in rows_for_display if row.get("sha256")}
    for row in pending_rows:
        preview_map[row["sha256"]] = row
    all_rows = load_emotion_rows(normalized_role)
    preview_rows = [preview_map[row.get("sha256", "")] for row in all_rows if row.get("sha256", "") in preview_map]
    summary = {
        "role": normalized_role,
        "model": model,
        "desc_prompt_path": str(DESCRIPTION_PROMPT_PATH),
        "record_path": str(jsonl_path),
        "total_asr_rows": total,
        "completed_before_run": total - len(pending_rows),
        "processed_this_run": processed_this_run,
        "error_count": error_count,
        "token_usage": usage_totals,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Emotion Desc] 运行完成。本次处理 {processed_this_run} 条。", flush=True)
    if usage_totals["total_tokens"] > 0:
        print(
            f"[Emotion Desc] API token: model={model} prompt={usage_totals['prompt_tokens']} "
            f"completion={usage_totals['completion_tokens']} total={usage_totals['total_tokens']}",
            flush=True,
        )
    print(f"[Emotion Desc] 运行记录: {jsonl_path}", flush=True)
    if not target_sha256:
        save_emotion_rows(normalized_role, preview_rows)
        print("[Emotion Desc] 全量运行结果已自动保存。", flush=True)
    else:
        print("[Emotion Desc] 本次结果仅供预览，点击\"保存情感标定\"后才会正式写入 workflow_db.csv。", flush=True)
    return preview_rows


def run_keyword_only(
    role: str,
    text_model: str = "",
    limit: int | None = None,
    sha256: str = "",
    force: bool = False,
) -> list[dict[str, Any]]:
    if not text_model.strip():
        raise ValueError("关键词提取模型不能为空。")
    normalized_role = normalize_role(role)
    target_sha256 = sha256.strip()
    kw_prompt = load_keyword_prompt()
    text_client = make_text_client(text_model)
    workflow_rows = [
        row
        for row in read_workflow_rows(normalized_role)
        if row.get("filter_status") == "confirmed" and row.get("asr_status") == "done" and row.get("emotion_description", "").strip()
    ]
    if target_sha256:
        if not any(row.get("sha256") == target_sha256 for row in workflow_rows):
            raise ValueError("所选音频尚未完成情感描述、未通过筛选或不存在，无法提取关键词。")
    preview_rows = load_emotion_rows(normalized_role)
    preview_by_hash = {row.get("sha256", ""): row for row in preview_rows if row.get("sha256")}
    rows_for_display: list[dict[str, Any]] = []
    pending_rows: list[dict[str, str]] = []
    for index, workflow_row in enumerate(sorted(workflow_rows, key=lambda row: (row.get("file_name", ""), row.get("sha256", ""))), start=1):
        view_row = {
            "sha256": workflow_row.get("sha256", ""),
            "索引": str(index),
            "语音文件": workflow_row.get("file_name", ""),
            "语音文本": workflow_row.get("asr_transcript", ""),
            "音频路径": str(
                resolve_audio_path_for_row(
                    role=workflow_row.get("role", "") or normalized_role,
                    audio_path=workflow_row.get("audio_path", ""),
                    stored_name=workflow_row.get("stored_name", ""),
                    file_name=workflow_row.get("file_name", ""),
                )
            ),
            "自然语言描述": workflow_row.get("emotion_description", ""),
            "情绪语气": workflow_row.get("emotion_tone", ""),
            "音频表达技巧": workflow_row.get("emotion_delivery", ""),
            "model": workflow_row.get("emotion_model", ""),
            "error": workflow_row.get("emotion_error", ""),
        }
        existing = preview_by_hash.get(view_row["sha256"])
        if existing is not None:
            view_row.update(existing)
        rows_for_display.append(view_row)
        should_run_row = not target_sha256 or workflow_row.get("sha256") == target_sha256
        has_description = bool(workflow_row.get("emotion_description", "").strip())
        has_keywords = bool(workflow_row.get("emotion_tone", "").strip() or workflow_row.get("emotion_delivery", "").strip())
        if should_run_row and has_description and (force or not has_keywords):
            pending_rows.append(view_row)
    if limit is not None:
        pending_rows = pending_rows[:limit]
    total = len(rows_for_display)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    records_dir = run_record_dir(normalized_role)
    records_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = records_dir / f"emotion_kw_{run_tag}.jsonl"
    summary_path = records_dir / f"emotion_kw_summary_{run_tag}.json"

    scope_text = "选中音频" if target_sha256 else "已有描述"
    run_total = len(pending_rows) if target_sha256 else total
    print(f"[Emotion KW] 开始运行，共 {run_total} 条 {scope_text}。", flush=True)
    print(f"[Emotion KW] 已完成 {total - len(pending_rows)} 条，待处理 {len(pending_rows)} 条。", flush=True)
    print(f"[Emotion KW] 文本模型={text_model}", flush=True)
    processed_this_run = 0
    error_count = 0
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with jsonl_path.open("w", encoding="utf-8") as log_file:
        for index, row in enumerate(pending_rows, start=1):
            voice_description = str(row.get(DESCRIPTION, "") or "").strip()
            if not voice_description:
                row["error"] = "关键词: 自然语言描述为空"
                error_count += 1
                log_file.write(
                    json.dumps(
                        {
                            "role": normalized_role,
                            "audio_file": row[AUDIO_FILE],
                            "transcript": row[TRANSCRIPT],
                            "text_model": text_model,
                            "kw_input": "",
                            "kw_output": "",
                            "kw_parsed_json": None,
                            "kw_usage": {},
                            "error": row["error"],
                            "preview_row": {
                                "sha256": row["sha256"],
                                KEYWORDS: row[KEYWORDS],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log_file.flush()
                processed_this_run += 1
                continue

            kw_result = None
            try:
                kw_result = call_keyword_model(text_client, kw_prompt, voice_description, text_model)
                row[TONE] = kw_result["tone_emotion_attitude"]
                row[DELIVERY] = kw_result["prosody_delivery"]
                row[KEYWORDS] = "，".join(part for part in [row[TONE], row[DELIVERY]] if part)
                row["error"] = ""
                add_usage(usage_totals, kw_result.get("usage", {}))
            except Exception as exc:
                kw_result = None
                row[TONE] = ""
                row[DELIVERY] = ""
                row[KEYWORDS] = ""
                row["error"] = f"关键词: {exc}"
                error_count += 1

            log_file.write(
                json.dumps(
                    {
                        "role": normalized_role,
                        "audio_file": row[AUDIO_FILE],
                        "transcript": row[TRANSCRIPT],
                        "text_model": text_model,
                        "voice_description": voice_description,
                        "kw_input": kw_result["input_text"] if kw_result is not None else "",
                        "kw_output": kw_result["output_text"] if kw_result is not None else "",
                        "kw_parsed_json": kw_result["parsed_json"] if kw_result is not None else None,
                        "kw_usage": kw_result.get("usage", {}) if kw_result is not None else {},
                        "error": row["error"],
                        "preview_row": {
                            "sha256": row["sha256"],
                            KEYWORDS: row[KEYWORDS],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log_file.flush()
            processed_this_run += 1
            if index % 5 == 0 or index == len(pending_rows):
                print(f"[Emotion KW] 已处理 {index}/{len(pending_rows)} 条。", flush=True)
            time.sleep(0.1)

    preview_map = {row.get("sha256", ""): row for row in rows_for_display if row.get("sha256")}
    for row in pending_rows:
        preview_map[row["sha256"]] = row
    all_rows = load_emotion_rows(normalized_role)
    preview_rows = [preview_map[row.get("sha256", "")] for row in all_rows if row.get("sha256", "") in preview_map]
    summary = {
        "role": normalized_role,
        "text_model": text_model,
        "kw_prompt_path": str(KEYWORD_PROMPT_PATH),
        "record_path": str(jsonl_path),
        "total_desc_rows": total,
        "completed_before_run": total - len(pending_rows),
        "processed_this_run": processed_this_run,
        "error_count": error_count,
        "token_usage": usage_totals,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Emotion KW] 运行完成。本次处理 {processed_this_run} 条。", flush=True)
    if usage_totals["total_tokens"] > 0:
        print(
            f"[Emotion KW] API token: model={text_model} prompt={usage_totals['prompt_tokens']} "
            f"completion={usage_totals['completion_tokens']} total={usage_totals['total_tokens']}",
            flush=True,
        )
    print(f"[Emotion KW] 运行记录: {jsonl_path}", flush=True)
    if not target_sha256:
        save_emotion_rows(normalized_role, preview_rows)
        print("[Emotion KW] 全量运行结果已自动保存。", flush=True)
    else:
        print("[Emotion KW] 本次结果仅供预览，点击\"保存情感标定\"后才会正式写入 workflow_db.csv。", flush=True)
    return preview_rows
