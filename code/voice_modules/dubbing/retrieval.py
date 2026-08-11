from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

from GUI.config import PROJECT_ROOT
from voice_modules.common.audio_manifest import resolve_audio_path_for_row
from voice_modules.common.workflow_db import AUDIO_FILE, AUDIO_PATH, IDX, KEYWORDS, LIB_DESC, TRANSCRIPT, read_workflow_rows
from voice_modules.role_library.role_library import RolePaths, resolve_role_paths


RERANK_MODEL = "qwen3-rerank"
TEXT_MODEL = "qwen3.6-flash"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
RERANK_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
DEFAULT_FINAL_RANKING_PROMPT_PATH = PROJECT_ROOT / "code" / "voice_modules" / "dubbing" / "prompts" / "prompt-最终排序.txt"

COL_INDEX = IDX
COL_AUDIO_FILE = AUDIO_FILE
COL_AUDIO_PATH = AUDIO_PATH
COL_TRANSCRIPT = TRANSCRIPT
COL_AUDIO_DESC = LIB_DESC
COL_KEYWORDS = KEYWORDS


@dataclass
class TextUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def get_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY。请先在设置页填写 Qwen API Key。")
    return api_key


def get_deepseek_api_key() -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY。请先在设置页填写 Deepseek API Key。")
    return api_key


def is_deepseek_model(model: str) -> bool:
    return model.strip().lower().startswith("deepseek")


def make_openai_client(model: str) -> OpenAI:
    if is_deepseek_model(model):
        return OpenAI(api_key=get_deepseek_api_key(), base_url=DEEPSEEK_BASE_URL)
    return OpenAI(api_key=get_api_key(), base_url=BASE_URL)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def read_text_file(path: Path) -> str:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise RuntimeError(f"无法读取文件: {path}")


def prepare_entry_records(role: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    normalized_role = resolve_role_paths(role).role
    workflow_rows = sorted(
        [
            item
            for item in read_workflow_rows(role)
            if item.get("filter_status") == "confirmed"
            and item.get("library_keywords", "").strip()
            and resolve_audio_path_for_row(
                role=item.get("role", "") or normalized_role,
                audio_path=item.get("audio_path", ""),
                stored_name=item.get("stored_name", ""),
                file_name=item.get("file_name", ""),
            ).exists()
        ],
        key=lambda row: (row.get("file_name", ""), row.get("sha256", "")),
    )
    for index, row in enumerate(
        workflow_rows,
        start=1,
    ):
        entries.append(
            {
                "entry_index": index,
                "audio_file": clean_text(row.get("file_name")),
                "audio_path": str(
                    resolve_audio_path_for_row(
                        role=row.get("role", "") or normalized_role,
                        audio_path=row.get("audio_path", ""),
                        stored_name=row.get("stored_name", ""),
                        file_name=row.get("file_name", ""),
                    )
                ),
                "duration_seconds": float(row.get("duration_seconds", 0) or 0),
                "transcript": clean_text(row.get("asr_transcript")),
                "asr_original_text": clean_text(row.get("asr_original_text")),
                "asr_language": clean_text(row.get("asr_language")),
                "audio_description": clean_text(row.get("library_description")),
                "emotion_tone": clean_text(row.get("emotion_tone")),
                "emotion_delivery": clean_text(row.get("emotion_delivery")),
                "keyword_text": clean_text(row.get("library_keywords")),
            }
        )
    return entries


def rerank_candidates(query_text: str, candidate_texts: list[str], top_n: int, rerank_model: str = RERANK_MODEL) -> dict[str, Any]:
    response = requests.post(
        RERANK_ENDPOINT,
        headers={"Authorization": f"Bearer {get_api_key()}", "Content-Type": "application/json"},
        json={
            "model": rerank_model,
            "query": query_text,
            "documents": candidate_texts,
            "top_n": top_n,
            "instruct": "Retrieve semantically similar emotion and tone descriptions.",
        },
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(f"第一层 rerank 请求失败: status={response.status_code} body={response.text}")
    data = response.json()
    if data.get("code"):
        raise RuntimeError(f"{data['code']}: {data.get('message', '')}")
    return data


def build_second_layer_candidate_block(entries: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for candidate_id, entry in enumerate(entries, start=1):
        blocks.append(f"候选{candidate_id}:")
        blocks.append(f"描述文本: {entry['audio_description']}")
        blocks.append(f"情绪语气关键词: {entry['keyword_text']}")
        blocks.append("")
    return "\n".join(blocks).strip()


def render_final_prompt(prompt_template: str, voice_description: str, entries: list[dict[str, Any]]) -> str:
    candidate_count = len(entries)
    example_entries = ", ".join(
        f'{{"candidate_id":{i},"rank":{i}}}' for i in range(1, min(candidate_count + 1, 6))
    )
    example_suffix = ",..." if candidate_count > 5 else ""
    replacements = {
        "{{输入查询}}": voice_description,
        "{{候选数量}}": str(candidate_count),
        "{{格式示例}}": f'{{"final_ranking":[{example_entries}{example_suffix}]}}',
        "{{候选列表}}": build_second_layer_candidate_block(entries),
    }
    rendered = prompt_template
    for placeholder, value in replacements.items():
        if placeholder not in rendered:
            raise ValueError(f"最终排序 prompt 缺少占位符: {placeholder}")
        rendered = rendered.replace(placeholder, value)
    return rendered


def generate_final_ranking(prompt: str, text_model: str = TEXT_MODEL) -> tuple[str, TextUsage]:
    request: dict[str, Any] = {
        "model": text_model,
        "messages": [
            {"role": "system", "content": "你是中文配音匹配排序助手，严格按要求输出，不做额外解释。"},
            {"role": "user", "content": prompt},
        ],
    }
    if not is_deepseek_model(text_model):
        request["extra_body"] = {"enable_thinking": False}
    response = make_openai_client(text_model).chat.completions.create(**request)
    content = clean_text(response.choices[0].message.content)
    if not content:
        raise RuntimeError("最终排序模型返回为空。")
    usage = getattr(response, "usage", None)
    return content, TextUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
    )


def parse_final_ranking_output(raw_output: str, candidate_count: int) -> list[int]:
    text = clean_text(raw_output)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    payload = json.loads(text)
    ranking = payload.get("final_ranking")
    if not isinstance(ranking, list) or len(ranking) != candidate_count:
        raise RuntimeError(f"最终排序数量不匹配: expected={candidate_count}, got={len(ranking) if isinstance(ranking, list) else 'invalid'}")
    ordered: list[tuple[int, int]] = []
    seen_ids: set[int] = set()
    seen_ranks: set[int] = set()
    for item in ranking:
        candidate_id = item.get("candidate_id")
        rank = item.get("rank")
        if not isinstance(candidate_id, int) or not isinstance(rank, int):
            raise RuntimeError("最终排序 JSON 中 candidate_id/rank 必须为整数。")
        if candidate_id in seen_ids or rank in seen_ranks:
            raise RuntimeError("最终排序 JSON 中存在重复 candidate_id 或 rank。")
        if not 1 <= candidate_id <= candidate_count or not 1 <= rank <= candidate_count:
            raise RuntimeError("最终排序 JSON 中 candidate_id/rank 超出范围。")
        seen_ids.add(candidate_id)
        seen_ranks.add(rank)
        ordered.append((rank, candidate_id))
    ordered.sort(key=lambda item: item[0])
    return [candidate_id for _, candidate_id in ordered]


def resolve_audio_path(role_paths: RolePaths, entry: dict[str, Any]) -> Path:
    raw_path = clean_text(entry.get("audio_path"))
    if raw_path:
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate.resolve()
    candidate = role_paths.audio_dir / Path(str(entry["audio_file"])).name
    return candidate.resolve()


def build_final_results(role_paths: RolePaths, ranked_entries: list[dict[str, Any]], final_order: list[int], top_k_final: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for final_rank, candidate_id in enumerate(final_order[:top_k_final], start=1):
        entry = dict(ranked_entries[candidate_id - 1])
        results.append(
            {
                "role": role_paths.role,
                "rank": final_rank,
                "candidate_id": candidate_id,
                COL_INDEX: entry["entry_index"],
                COL_AUDIO_FILE: entry["audio_file"],
                "audio_path": str(resolve_audio_path(role_paths, entry)),
                "duration_seconds": entry["duration_seconds"],
                COL_TRANSCRIPT: entry["transcript"],
                "asr_original_text": entry["asr_original_text"],
                "asr_language": entry["asr_language"],
                COL_AUDIO_DESC: entry["audio_description"],
                "情绪语气": entry["emotion_tone"],
                "语音技巧": entry["emotion_delivery"],
                "音频表达技巧": entry["emotion_delivery"],
                COL_KEYWORDS: entry["keyword_text"],
                "rerank_rank": entry["rerank_rank"],
                "rerank_score": entry["rerank_score"],
            }
        )
    return results


def run_double_keyword_retrieval(
    *,
    role: str,
    transcript_text: str,
    voice_description: str,
    top_k_second: int = 10,
    top_k_final: int = 10,
    rerank_model: str = RERANK_MODEL,
    text_model: str = TEXT_MODEL,
    final_ranking_prompt_path: Path = DEFAULT_FINAL_RANKING_PROMPT_PATH,
) -> dict[str, Any]:
    role_paths = resolve_role_paths(role)
    entries = prepare_entry_records(role_paths.role)
    if not entries:
        raise RuntimeError("workflow_db.csv 中没有可检索条目。")

    rerank_data = rerank_candidates(
        query_text=voice_description,
        candidate_texts=[entry["keyword_text"] for entry in entries],
        top_n=min(top_k_second, len(entries)),
        rerank_model=rerank_model,
    )
    rerank_results = rerank_data.get("results")
    if rerank_results is None:
        rerank_results = (rerank_data.get("output", {}) or {}).get("results", [])
    if not rerank_results:
        raise RuntimeError("第一层 rerank 没有返回候选。")

    second_layer_entries: list[dict[str, Any]] = []
    for rerank_rank, item in enumerate(rerank_results, start=1):
        entry = dict(entries[int(item["index"])])
        entry["rerank_rank"] = rerank_rank
        entry["rerank_score"] = float(item["relevance_score"])
        second_layer_entries.append(entry)

    final_prompt = render_final_prompt(read_text_file(final_ranking_prompt_path), voice_description, second_layer_entries)
    second_layer_raw_json, second_usage = generate_final_ranking(final_prompt, text_model=text_model)
    final_order = parse_final_ranking_output(second_layer_raw_json, len(second_layer_entries))
    top_results = build_final_results(role_paths, second_layer_entries, final_order, min(top_k_final, len(second_layer_entries)))
    rerank_usage = rerank_data.get("usage", {}) or {}
    model_token_usage = {
        "rerank": {
            "model": rerank_model,
            "input_tokens": int(rerank_usage.get("input_tokens", 0) or 0),
            "total_tokens": int(rerank_usage.get("total_tokens", 0) or 0),
        },
        "second_layer": {
            "model": text_model,
            "prompt_tokens": second_usage.prompt_tokens,
            "completion_tokens": second_usage.completion_tokens,
            "total_tokens": second_usage.total_tokens,
        },
    }

    return {
        "role": role_paths.role,
        "role_audio_dir": str(role_paths.audio_dir),
        "workflow_db_path": str(role_paths.excel_path),
        "input_transcript": transcript_text,
        "voice_description": voice_description,
        "final_ranking_prompt": str(final_ranking_prompt_path),
        "indexed_entry_count": len(entries),
        "second_layer_candidate_count": len(second_layer_entries),
        "model_token_usage": model_token_usage,
        "second_layer_raw_json": second_layer_raw_json,
        "top_results": top_results,
    }
