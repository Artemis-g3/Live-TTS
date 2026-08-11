from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from bailian_retrieval_common import (
    COL_AUDIO_DESC,
    COL_AUDIO_FILE,
    COL_INDEX,
    COL_KEYWORDS,
    COL_TRANSCRIPT,
    DEFAULT_FINAL_RANKING_PROMPT_PATH,
    RERANK_MODEL,
    TEXT_MODEL,
    generate_text_from_user_prompt,
    load_excel_records,
    make_openai_client,
    parse_final_ranking_output,
    prepare_entry_records,
    read_text_file,
    render_prompt_template,
    rerank_candidates,
)
from role_data import RolePaths, ensure_role_excel

DEFAULT_TOP_K_SECOND = 10
DEFAULT_TOP_K_FINAL = 10


def read_voice_description(arg_text: str | None) -> str:
    if arg_text and arg_text.strip():
        return arg_text.strip()
    text = input("请输入希望的配音描述: ").strip()
    if not text:
        raise ValueError("配音描述不能为空。")
    return text


def resolve_audio_path(role_paths: RolePaths, entry: dict[str, Any]) -> Path:
    raw_audio_path = str(entry.get("audio_path", "") or "").strip()
    if raw_audio_path:
        candidate = Path(raw_audio_path)
        if candidate.exists():
            return candidate.resolve()

    candidate = Path(str(entry["audio_file"]))
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    return (role_paths.audio_dir / candidate.name).resolve()


def build_second_layer_candidate_block(entries: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for candidate_id, entry in enumerate(entries, start=1):
        blocks.append(f"候选{candidate_id}:")
        blocks.append(f"描述文本: {entry['audio_description']}")
        blocks.append(f"情绪语气关键词: {entry['keyword_text']}")
        blocks.append("")
    return "\n".join(blocks).strip()


def build_final_results(
    *,
    role_paths: RolePaths,
    ranked_entries: list[dict[str, Any]],
    final_order: list[int],
    top_k_final: int,
) -> list[dict[str, Any]]:
    final_results: list[dict[str, Any]] = []
    for final_rank, candidate_id in enumerate(final_order[:top_k_final], start=1):
        entry = dict(ranked_entries[candidate_id - 1])
        audio_path = str(resolve_audio_path(role_paths, entry))
        final_results.append(
            {
                "role": role_paths.role,
                "rank": final_rank,
                "candidate_id": candidate_id,
                COL_INDEX: entry["entry_index"],
                COL_AUDIO_FILE: entry["audio_file"],
                "audio_path": audio_path,
                COL_TRANSCRIPT: entry["transcript"],
                COL_AUDIO_DESC: entry["audio_description"],
                COL_KEYWORDS: entry["keyword_text"],
                "rerank_rank": entry["rerank_rank"],
                "rerank_score": entry["rerank_score"],
            }
        )
    return final_results


def run_double_keyword_retrieval(
    *,
    role: str,
    transcript_text: str,
    voice_description: str,
    excel_path: Path | None = None,
    final_ranking_prompt_path: Path = DEFAULT_FINAL_RANKING_PROMPT_PATH,
    top_k_second: int = DEFAULT_TOP_K_SECOND,
    top_k_final: int = DEFAULT_TOP_K_FINAL,
) -> dict[str, Any]:
    role_paths = ensure_role_excel(role)
    if excel_path is None:
        excel_path = role_paths.excel_path
    excel_path = excel_path.resolve()

    raw_df = load_excel_records(excel_path)
    entries = prepare_entry_records(raw_df)
    if not entries:
        raise RuntimeError("Excel 中没有可用于检索的条目。")

    rerank_data = rerank_candidates(
        query_text=voice_description,
        candidate_texts=[entry["keyword_text"] for entry in entries],
        top_n=min(top_k_second, len(entries)),
    )
    rerank_usage = rerank_data.get("usage", {}) or {}
    rerank_results = rerank_data.get("results")
    if rerank_results is None:
        rerank_results = (rerank_data.get("output", {}) or {}).get("results", [])
    if not rerank_results:
        raise RuntimeError("First-layer rerank returned no candidates.")

    second_layer_entries: list[dict[str, Any]] = []
    for rerank_rank, item in enumerate(rerank_results, start=1):
        entry = dict(entries[int(item["index"])])
        entry["rerank_rank"] = rerank_rank
        entry["rerank_score"] = float(item["relevance_score"])
        second_layer_entries.append(entry)

    final_prompt = read_text_file(final_ranking_prompt_path)
    final_prompt_rendered = render_prompt_template(
        prompt_template=final_prompt,
        replacements={
            "{{输入查询}}": voice_description,
            "{{候选列表}}": build_second_layer_candidate_block(second_layer_entries),
        },
    )
    client = make_openai_client()
    second_layer_raw_json, second_layer_usage = generate_text_from_user_prompt(
        client=client,
        user_prompt=final_prompt_rendered,
        model=TEXT_MODEL,
    )
    final_order = parse_final_ranking_output(
        raw_output=second_layer_raw_json,
        candidate_count=len(second_layer_entries),
    )
    top_results = build_final_results(
        role_paths=role_paths,
        ranked_entries=second_layer_entries,
        final_order=final_order,
        top_k_final=min(top_k_final, len(second_layer_entries)),
    )

    return {
        "role": role_paths.role,
        "role_audio_dir": str(role_paths.audio_dir),
        "role_excel_path": str(excel_path),
        "input_transcript": transcript_text,
        "voice_description": voice_description,
        "excel_path": str(excel_path),
        "final_ranking_prompt": str(final_ranking_prompt_path),
        "indexed_entry_count": len(entries),
        "second_layer_candidate_count": len(second_layer_entries),
        "model_token_usage": {
            "rerank": {
                "model": RERANK_MODEL,
                "input_tokens": int(rerank_usage.get("input_tokens", 0) or 0),
                "total_tokens": int(rerank_usage.get("total_tokens", 0) or 0),
            },
            "second_layer": {
                "model": TEXT_MODEL,
                "prompt_tokens": second_layer_usage.prompt_tokens,
                "completion_tokens": second_layer_usage.completion_tokens,
                "total_tokens": second_layer_usage.total_tokens,
            },
        },
        "second_layer_raw_json": second_layer_raw_json,
        "top_results": top_results,
    }


def print_retrieval_result(result: dict[str, Any]) -> None:
    print(f"role={result['role']}")
    print(f"role_audio_dir={result['role_audio_dir']}")
    print(f"role_excel_path={result['role_excel_path']}")
    print(f"input_transcript={result['input_transcript']}")
    print(f"voice_description={result['voice_description']}")
    print(f"excel_path={result['excel_path']}")
    print(f"final_ranking_prompt={result['final_ranking_prompt']}")
    print(f"indexed_entry_count={result['indexed_entry_count']}")
    print(f"second_layer_candidate_count={result['second_layer_candidate_count']}")
    print("model_token_usage:")

    rerank_usage = result["model_token_usage"]["rerank"]
    print(
        f"- {rerank_usage['model']}(first_layer): "
        f"input={rerank_usage['input_tokens']} total={rerank_usage['total_tokens']}"
    )

    second_layer_usage = result["model_token_usage"]["second_layer"]
    print(
        f"- {second_layer_usage['model']}(second_layer): "
        f"prompt={second_layer_usage['prompt_tokens']} "
        f"completion={second_layer_usage['completion_tokens']} "
        f"total={second_layer_usage['total_tokens']}"
    )

    print("second_layer_raw_json:")
    print(result["second_layer_raw_json"])
    print("top_results:")
    for row in result["top_results"]:
        print(
            f"[final={row['rank']}] [candidate={row['candidate_id']}] "
            f"[rerank={row['rerank_rank']}] rerank_score={row['rerank_score']:.6f} "
            f"entry_index={row[COL_INDEX]} audio_file={row[COL_AUDIO_FILE]}"
        )
        print(f"语音文本: {row[COL_TRANSCRIPT]}")
        print(f"语音描述: {row[COL_AUDIO_DESC]}")
        print(f"原始关键词: {row[COL_KEYWORDS]}")
        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="按角色执行 rerank + 大模型二层检索。")
    parser.add_argument("--role", required=True, help="角色名，例如 莫宁 或 尤诺")
    parser.add_argument("--transcript-text", default="", help="待合成的配音台词，只用于输出记录")
    parser.add_argument("--voice-description", default=None, help="用于检索排序的希望配音描述")
    parser.add_argument("--excel", default=None, help="可选，覆盖角色默认 Excel 路径")
    parser.add_argument(
        "--final-ranking-prompt",
        default=str(DEFAULT_FINAL_RANKING_PROMPT_PATH),
        help="最终排序 prompt 模板路径",
    )
    parser.add_argument("--top-k-second", type=int, default=DEFAULT_TOP_K_SECOND, help="第二层候选数量")
    parser.add_argument("--top-k-final", type=int, default=DEFAULT_TOP_K_FINAL, help="最终输出数量")
    args = parser.parse_args()

    result = run_double_keyword_retrieval(
        role=args.role,
        transcript_text=args.transcript_text.strip(),
        voice_description=read_voice_description(args.voice_description),
        excel_path=Path(args.excel).resolve() if args.excel else None,
        final_ranking_prompt_path=Path(args.final_ranking_prompt).resolve(),
        top_k_second=args.top_k_second,
        top_k_final=args.top_k_final,
    )
    print_retrieval_result(result)


if __name__ == "__main__":
    main()
