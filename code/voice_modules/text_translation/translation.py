from __future__ import annotations

from pathlib import Path
from typing import Any

from GUI.config import PROJECT_ROOT
from voice_modules.common.workflow_db import ASR_CHINESE_TEXT, ASR_ERROR, ASR_FILE, ASR_LANGUAGE, ASR_ORIGINAL_TEXT, TRANSLATION_ERROR
from voice_modules.dubbing.retrieval import clean_text, is_deepseek_model, make_openai_client, read_text_file


PROMPT_PATH = PROJECT_ROOT / "code" / "voice_modules" / "text_translation" / "prompts" / "translate_to_zh_cn.txt"


def load_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"未找到翻译提示词文件: {PROMPT_PATH}")
    prompt = read_text_file(PROMPT_PATH).strip()
    if not prompt:
        raise ValueError(f"翻译提示词文件为空: {PROMPT_PATH}")
    return prompt


def render_prompt(prompt: str, source_language: str, original_text: str) -> str:
    rendered = prompt.replace("{{源语言}}", source_language).replace("{{原始文本}}", original_text.strip())
    return rendered


def translate_once(source_language: str, original_text: str, model: str) -> str:
    prompt = render_prompt(load_prompt(), source_language, original_text)
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个严格的翻译助手，只输出简体中文译文，不做解释。"},
            {"role": "user", "content": prompt},
        ],
    }
    if not is_deepseek_model(model):
        request["extra_body"] = {"enable_thinking": False}
    response = make_openai_client(model).chat.completions.create(**request)
    translation = clean_text(response.choices[0].message.content)
    if not translation:
        raise RuntimeError("翻译模型返回为空。")
    return translation


def run_translation_preview(
    *,
    role: str,
    rows: list[dict[str, Any]],
    language: str,
    model: str,
    sha256: str = "",
    force: bool = False,
) -> list[dict[str, Any]]:
    del role
    target_sha256 = sha256.strip()
    updated_rows = [dict(row) for row in rows]
    pending_rows = [
        row
        for row in updated_rows
        if str(row.get("sha256", "") or "").strip()
        and (not target_sha256 or str(row.get("sha256", "") or "").strip() == target_sha256)
        and (force or not str(row.get(ASR_CHINESE_TEXT, "") or "").strip())
    ]
    if target_sha256 and not pending_rows:
        raise ValueError("所选音频不存在，无法运行单条翻译。")

    print(f"[Translate] 开始运行，共 {len(pending_rows)} 条。", flush=True)
    print(f"[Translate] 原始语言={language}", flush=True)
    for index, row in enumerate(pending_rows, start=1):
        original_text = str(row.get(ASR_ORIGINAL_TEXT, "") or "").strip()
        row[ASR_LANGUAGE] = language
        if not original_text:
            row[ASR_CHINESE_TEXT] = ""
            row["translation_model"] = ""
            row[TRANSLATION_ERROR] = "原始文本为空，无法翻译。"
            continue
        try:
            if language == "zh":
                row[ASR_CHINESE_TEXT] = original_text
                row["translation_model"] = "direct_copy"
            else:
                row[ASR_CHINESE_TEXT] = translate_once(language, original_text, model)
                row["translation_model"] = model
            row[TRANSLATION_ERROR] = ""
        except Exception as exc:
            row[ASR_CHINESE_TEXT] = ""
            row["translation_model"] = model if language != "zh" else "direct_copy"
            row[TRANSLATION_ERROR] = str(exc)
        if index % 5 == 0 or index == len(pending_rows):
            print(f"[Translate] 已处理 {index}/{len(pending_rows)} 条。", flush=True)
    if not target_sha256:
        from voice_modules.speech_recognition.asr import save_asr_rows
        save_asr_rows(role, updated_rows)
        print("[Translate] 全量翻译结果已自动保存。", flush=True)
    else:
        print("[Translate] 本次结果仅供预览，点击“全部保存”后才会正式写入 workflow_db.csv。", flush=True)
    print("[Translate] 运行完成。", flush=True)
    return updated_rows
