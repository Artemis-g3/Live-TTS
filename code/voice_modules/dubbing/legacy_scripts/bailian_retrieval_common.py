from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-v4"
TEXT_MODEL = "qwen3.6-flash"
RERANK_MODEL = "qwen3-rerank"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
RERANK_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"

DEFAULT_RERANK_INSTRUCT = "Retrieve semantically similar emotion and tone descriptions."
TEXT_SYSTEM_PROMPT = "你是中文配音匹配排序助手，严格按要求输出，不做额外解释。"

DEFAULT_EMBEDDING_DIMENSIONS = 1024
MAX_EMBEDDING_BATCH = 10
KEYWORD_SEPARATOR = "，"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_INDEX_DIR = DEFAULT_OUTPUT_DIR / "index_artifacts"
DEFAULT_EXCEL_PATH = PROJECT_ROOT / "情绪关键词.xlsx"
DEFAULT_INPUT_TO_DESC_PROMPT_PATH = PROJECT_ROOT / "prompt-输入到描述.txt"
DEFAULT_DESC_TO_KEYWORDS_PROMPT_PATH = PROJECT_ROOT / "prompt-描述到关键词.txt"
DEFAULT_FINAL_RANKING_PROMPT_PATH = PROJECT_ROOT / "prompts" / "prompt-最终排序.txt"

COL_INDEX = "索引"
COL_AUDIO_FILE = "语音文件"
COL_AUDIO_PATH = "音频路径"
COL_TRANSCRIPT = "语音文本"
COL_AUDIO_DESC = "语音描述-音频理解模型"
COL_KEYWORDS = "关键词"

COL_MATCHED_KEYWORD = "matched_keyword"
COL_MATCHED_KEYWORDS = "matched_keywords"
COL_MATCHED_USER_KEYWORDS = "matched_user_keywords"
COL_BEST_KEYWORD_MATCH = "best_keyword_match"
COL_EMBEDDING_SCORE = "embedding_score"
COL_FINAL_KEYWORD_SCORE = "final_keyword_score"


@dataclass
class EmbeddingUsage:
    input_tokens: int = 0
    total_tokens: int = 0

    def add_response(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        total = getattr(usage, "total_tokens", None)
        if total is None:
            total = getattr(usage, "prompt_tokens", 0) or 0
        self.total_tokens += int(total)


@dataclass
class TextGenerationUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add_response(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        total = getattr(usage, "total_tokens", None)
        if total is None:
            total = (getattr(usage, "prompt_tokens", 0) or 0) + (
                getattr(usage, "completion_tokens", 0) or 0
            )
        self.total_tokens += int(total)


def get_api_key() -> str:
    env_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not env_key:
        raise RuntimeError("Missing DashScope API key. Set DASHSCOPE_API_KEY.")
    return env_key


def make_openai_client() -> OpenAI:
    return OpenAI(api_key=get_api_key(), base_url=BASE_URL)


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text_file(path: Path, encodings: Iterable[str] = ("utf-8", "utf-8-sig", "gb18030")) -> str:
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to read file: {path}")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def load_excel_records(excel_path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    required_columns = {COL_INDEX, COL_AUDIO_FILE, COL_TRANSCRIPT, COL_AUDIO_DESC, COL_KEYWORDS}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Excel 缺少字段: {sorted(missing)}")
    return df


def split_keywords(keyword_text: str) -> list[str]:
    normalized = clean_text(keyword_text)
    if not normalized:
        return []
    for separator in ["；", ";", "、", "|", "/", "，", ",", "\n", "\r"]:
        normalized = normalized.replace(separator, "||")
    parts = [item.strip() for item in normalized.split("||")]
    return [item for item in parts if item]


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        value = clean_text(item)
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def prepare_keyword_index_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    records: list[dict[str, Any]] = []
    skipped_count = 0
    for _, row in df.iterrows():
        source_keywords = clean_text(row[COL_KEYWORDS])
        audio_desc = clean_text(row[COL_AUDIO_DESC])
        transcript = clean_text(row[COL_TRANSCRIPT])
        audio_file = clean_text(row[COL_AUDIO_FILE])
        keywords = split_keywords(source_keywords)
        if not keywords:
            skipped_count += 1
            continue
        base_record = {
            COL_INDEX: int(row[COL_INDEX]),
            COL_AUDIO_FILE: audio_file,
            COL_TRANSCRIPT: transcript,
            COL_AUDIO_DESC: audio_desc,
            COL_KEYWORDS: source_keywords,
        }
        for keyword in keywords:
            item = dict(base_record)
            item[COL_MATCHED_KEYWORD] = keyword
            records.append(item)
    return pd.DataFrame(records), skipped_count


def prepare_entry_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        entries.append(
            {
                "entry_index": int(row[COL_INDEX]),
                "audio_file": clean_text(row[COL_AUDIO_FILE]),
                "audio_path": clean_text(row[COL_AUDIO_PATH]) if COL_AUDIO_PATH in row.index else "",
                "transcript": clean_text(row[COL_TRANSCRIPT]),
                "audio_description": clean_text(row[COL_AUDIO_DESC]),
                "keyword_text": clean_text(row[COL_KEYWORDS]),
            }
        )
    return entries


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def embed_texts(
    client: OpenAI,
    texts: list[str],
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    batch_size: int = MAX_EMBEDDING_BATCH,
) -> tuple[np.ndarray, EmbeddingUsage]:
    if batch_size < 1 or batch_size > MAX_EMBEDDING_BATCH:
        raise ValueError(f"batch_size must be between 1 and {MAX_EMBEDDING_BATCH}")
    if not texts:
        raise ValueError("No texts to embed.")

    usage = EmbeddingUsage()
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=dimensions,
            encoding_format="float",
        )
        usage.add_response(response)
        batch_vectors = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        chunks.append(batch_vectors)
    vectors = np.vstack(chunks)
    return l2_normalize(vectors).astype(np.float32), usage


def embed_query(
    client: OpenAI,
    text: str,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> tuple[np.ndarray, EmbeddingUsage]:
    vectors, usage = embed_texts(client=client, texts=[text], dimensions=dimensions, batch_size=1)
    return vectors[0], usage


def generate_text(
    client: OpenAI,
    prompt_template: str,
    content_text: str,
    placeholder: str,
    model: str = TEXT_MODEL,
) -> tuple[str, TextGenerationUsage]:
    if placeholder not in prompt_template:
        raise ValueError(f"Prompt template missing placeholder: {placeholder}")
    user_prompt = prompt_template.replace(placeholder, content_text.strip())
    return generate_text_from_user_prompt(client=client, user_prompt=user_prompt, model=model)


def generate_text_from_user_prompt(
    client: OpenAI,
    user_prompt: str,
    model: str = TEXT_MODEL,
) -> tuple[str, TextGenerationUsage]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TEXT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        extra_body={"enable_thinking": False},
    )
    usage = TextGenerationUsage()
    usage.add_response(response)
    result = clean_text(response.choices[0].message.content)
    if not result:
        raise RuntimeError("Model returned empty text.")
    return result, usage


def extract_keywords_from_model_output(raw_output: str) -> list[str]:
    lines = [clean_text(line) for line in clean_text(raw_output).splitlines() if clean_text(line)]
    text = lines[-1] if lines else clean_text(raw_output)
    prefixes = ("关键词：", "关键词:", "关键信息：", "关键信息:")
    for prefix in prefixes:
        if text.startswith(prefix):
            text = clean_text(text[len(prefix) :])
            break
    text = text.replace("、", KEYWORD_SEPARATOR).replace(",", KEYWORD_SEPARATOR)
    return unique_preserve_order(text.split(KEYWORD_SEPARATOR))


def manifest_path(index_dir: Path) -> Path:
    return index_dir / "keyword_embedding_manifest.json"


def metadata_path(index_dir: Path) -> Path:
    return index_dir / "keyword_embedding_metadata.csv"


def vectors_path(index_dir: Path) -> Path:
    return index_dir / "keyword_embedding_vectors.npy"


def save_manifest(index_dir: Path, payload: dict[str, Any]) -> None:
    manifest_path(index_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_manifest(index_dir: Path) -> dict[str, Any]:
    return json.loads(manifest_path(index_dir).read_text(encoding="utf-8"))


def render_prompt_template(prompt_template: str, replacements: dict[str, str]) -> str:
    rendered = prompt_template
    for placeholder, value in replacements.items():
        if placeholder not in rendered:
            raise ValueError(f"Prompt template missing placeholder: {placeholder}")
        rendered = rendered.replace(placeholder, clean_text(value))
    return rendered


def parse_final_ranking_output(raw_output: str, candidate_count: int) -> list[int]:
    text = clean_text(raw_output)
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Final ranking output is not valid JSON: {exc}") from exc

    final_ranking = payload.get("final_ranking")
    if not isinstance(final_ranking, list):
        raise RuntimeError("Final ranking JSON is missing the final_ranking array.")
    if len(final_ranking) != candidate_count:
        raise RuntimeError(
            f"Final ranking candidate count mismatch: expected {candidate_count}, got {len(final_ranking)}"
        )

    seen_ids: set[int] = set()
    seen_ranks: set[int] = set()
    ordered: list[tuple[int, int]] = []
    for item in final_ranking:
        if not isinstance(item, dict):
            raise RuntimeError("Each final_ranking item must be an object.")
        candidate_id = item.get("candidate_id")
        rank = item.get("rank")
        if not isinstance(candidate_id, int) or not isinstance(rank, int):
            raise RuntimeError("candidate_id and rank must both be integers.")
        if not 1 <= candidate_id <= candidate_count:
            raise RuntimeError(f"candidate_id out of range: {candidate_id}")
        if not 1 <= rank <= candidate_count:
            raise RuntimeError(f"rank out of range: {rank}")
        if candidate_id in seen_ids:
            raise RuntimeError(f"Duplicate candidate_id in final ranking: {candidate_id}")
        if rank in seen_ranks:
            raise RuntimeError(f"Duplicate rank in final ranking: {rank}")
        seen_ids.add(candidate_id)
        seen_ranks.add(rank)
        ordered.append((rank, candidate_id))

    ordered.sort(key=lambda item: item[0])
    return [candidate_id for _, candidate_id in ordered]


def rerank_candidates(
    query_text: str,
    candidate_texts: list[str],
    top_n: int,
    instruct: str = DEFAULT_RERANK_INSTRUCT,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": RERANK_MODEL,
        "query": query_text,
        "documents": candidate_texts,
        "top_n": top_n,
        "instruct": instruct,
    }
    response = requests.post(RERANK_ENDPOINT, headers=headers, json=payload, timeout=120)
    if not response.ok:
        raise RuntimeError(
            f"Rerank request failed: status={response.status_code} body={response.text}"
        )
    data = response.json()
    if data.get("code"):
        raise RuntimeError(f"{data['code']}: {data.get('message', '')}")
    return data


def validate_index_artifacts(index_dir: Path, excel_path: Path) -> dict[str, Any]:
    missing = [
        str(path.name)
        for path in [manifest_path(index_dir), metadata_path(index_dir), vectors_path(index_dir)]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"索引文件缺失: {missing}")

    manifest = load_manifest(index_dir)
    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        raise RuntimeError(
            f"索引模型不匹配: {manifest.get('embedding_model')} != {EMBEDDING_MODEL}"
        )

    current_excel = excel_path.resolve()
    expected_excel = Path(manifest["source_excel"]).resolve()
    if current_excel != expected_excel:
        if current_excel.name == expected_excel.name:
            expected_excel = current_excel
        else:
            raise RuntimeError(f"索引来源不匹配: {current_excel} != {expected_excel}")

    current_mtime = current_excel.stat().st_mtime
    if current_mtime > float(manifest.get("source_excel_mtime", 0)):
        raise RuntimeError("Excel 已更新，现有关键词索引已过期，请先重建关键词向量索引。")

    return manifest
