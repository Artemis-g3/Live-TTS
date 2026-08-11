from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from voice_modules.common.audio_manifest import move_rows_to_trash, read_manifest, rows_as_dicts, set_manifest_status
from voice_modules.input_audio.input_audio import duration_stats, scan_input_audio, sync_input_audio
from voice_modules.common.project_state import get_project_state
from voice_modules.common.result import BackendResult
from voice_modules.common.workspace_migration import migrate_workspace
from voice_modules.common.workflow_db import ensure_workflow_db
from GUI.config import AppConfig, PROJECT_ROOT, to_jsonable


RESULT_PREFIX = "VOICE_GUI_RESULT_JSON="


def emit_result(result: BackendResult) -> None:
    print(RESULT_PREFIX + json.dumps(to_jsonable(result.to_dict()), ensure_ascii=False), flush=True)


def read_json(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def configure_env(args: argparse.Namespace) -> None:
    api_key = getattr(args, "api_key", "") or ""
    if api_key.strip():
        os.environ["DASHSCOPE_API_KEY"] = api_key.strip()
    deepseek_api_key = getattr(args, "deepseek_api_key", "") or ""
    if deepseek_api_key.strip():
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key.strip()


def command_state(_: argparse.Namespace) -> BackendResult:
    return BackendResult(success=True, message="项目状态扫描完成。", data=get_project_state())


def command_migrate_workspace(_: argparse.Namespace) -> BackendResult:
    return BackendResult(success=True, message="workspace 迁移完成。", data=migrate_workspace())


def command_import_role_from_excel(args: argparse.Namespace) -> BackendResult:
    from voice_modules.common.legacy_excel_import import ImportValidationError, import_role_from_excel

    print("[Import] 未运行筛选 / 未运行 ASR / 未运行情感标定，仅导入历史结果。", flush=True)
    try:
        summary = import_role_from_excel(
            role=args.role,
            audio_dir=Path(args.audio_dir).resolve(),
            excel_path=Path(args.excel).resolve(),
        )
    except ImportValidationError as exc:
        return BackendResult(
            success=False,
            message=exc.message,
            data={
                "role": args.role,
                "mismatch_count": exc.mismatch_count,
                "sample_mismatches": list(exc.sample_mismatches),
            },
        )
    return BackendResult(
        success=True,
        message="历史角色数据导入完成。",
        output_paths={
            "workflow_db_path": summary["workflow_db_path"],
            "library_excel_path": summary["library_excel_path"],
        },
        data=summary,
    )


def command_sync_workflow_db(args: argparse.Namespace) -> BackendResult:
    result = ensure_workflow_db(args.role)
    return BackendResult(
        success=True,
        message="统一主表同步完成。",
        output_paths={"workflow_db": str(result.path)},
        data={"role": result.role, "row_count": result.row_count},
    )


def command_scan_input_audio(args: argparse.Namespace) -> BackendResult:
    rows = scan_input_audio(args.role)
    return BackendResult(success=True, message="输入音频扫描完成。", data={"rows": rows})


def command_sync_input_audio(args: argparse.Namespace) -> BackendResult:
    rows = sync_input_audio(args.role)
    return BackendResult(success=True, message="输入音频同步完成。", data={"rows": rows})


def command_duration_stats(args: argparse.Namespace) -> BackendResult:
    return BackendResult(success=True, message="时长分布统计完成。", data=duration_stats(args.role))


def command_load_manifest(args: argparse.Namespace) -> BackendResult:
    return BackendResult(success=True, message="音频清单加载完成。", data={"rows": rows_as_dicts(read_manifest(args.role))})


def command_update_filter_status(args: argparse.Namespace) -> BackendResult:
    sha_values = read_json(args.input_json)
    if not isinstance(sha_values, list):
        raise ValueError("input-json 必须是 sha256 数组。")
    if args.status == "deleted":
        rows = move_rows_to_trash(args.role, [str(value) for value in sha_values])
    else:
        rows = set_manifest_status(args.role, [str(value) for value in sha_values], filter_status=args.status)
    return BackendResult(success=True, message="音频状态已更新。", data={"rows": rows_as_dicts(rows)})


def command_update_reference(args: argparse.Namespace) -> BackendResult:
    sha_values = read_json(args.input_json)
    if not isinstance(sha_values, list):
        raise ValueError("input-json 必须是 sha256 数组。")
    rows = set_manifest_status(args.role, [str(value) for value in sha_values], selected_reference=args.selected)
    return BackendResult(success=True, message="参考音频标记已更新。", data={"rows": rows_as_dicts(rows)})


def command_save_reference_selection(args: argparse.Namespace) -> BackendResult:
    sha_values = read_json(args.input_json)
    if not isinstance(sha_values, list):
        raise ValueError("input-json 必须是 sha256 数组。")
    all_sha_values = [row.sha256 for row in read_manifest(args.role) if row.sha256]
    if all_sha_values:
        set_manifest_status(args.role, all_sha_values, selected_reference=False)
    rows = set_manifest_status(args.role, [str(value) for value in sha_values], selected_reference=True)
    return BackendResult(success=True, message="参考音频选择已保存。", data={"rows": rows_as_dicts(rows)})


def command_filter_audio(args: argparse.Namespace) -> BackendResult:
    from voice_modules.audio_filter.speaker_filter import run_speaker_filter

    data = run_speaker_filter(args.role, args.confirm_threshold, args.review_threshold)
    return BackendResult(success=True, message="音频筛选完成，请确认后点击保存结果。", data=data)


def command_save_filter_results(args: argparse.Namespace) -> BackendResult:
    from voice_modules.audio_filter.speaker_filter import save_filter_results

    rows = read_json(args.input_json)
    if not isinstance(rows, list):
        raise ValueError("筛选保存数据必须是数组。")
    saved_rows = save_filter_results(args.role, rows)
    return BackendResult(success=True, message="筛选结果已保存。", data={"rows": saved_rows})


def command_load_asr(args: argparse.Namespace) -> BackendResult:
    from voice_modules.speech_recognition.asr import load_asr_rows

    return BackendResult(success=True, message="ASR 结果加载完成。", data={"rows": load_asr_rows(args.role)})


def command_run_asr(args: argparse.Namespace) -> BackendResult:
    from voice_modules.speech_recognition.asr import run_asr

    rows = run_asr(
        args.role,
        model=args.model,
        language=args.language,
        limit=args.limit if args.limit > 0 else None,
        sha256=args.sha256,
        force=args.force,
    )
    return BackendResult(success=True, message="语音识别完成。", data={"rows": rows})


def command_save_asr(args: argparse.Namespace) -> BackendResult:
    from voice_modules.speech_recognition.asr import save_asr_rows

    rows = read_json(args.input_json)
    if not isinstance(rows, list):
        raise ValueError("ASR 保存数据必须是数组。")
    path = save_asr_rows(args.role, rows)
    return BackendResult(success=True, message="ASR 与翻译结果已保存。", output_paths={"workflow_db": str(path)}, data={"row_count": len(rows)})


def command_run_translation(args: argparse.Namespace) -> BackendResult:
    from voice_modules.text_translation.translation import run_translation_preview

    rows = read_json(args.input_json)
    if not isinstance(rows, list):
        raise ValueError("翻译输入数据必须是数组。")
    preview_rows = run_translation_preview(
        role=args.role,
        rows=rows,
        language=args.language,
        model=args.model,
        sha256=args.sha256,
        force=args.force,
    )
    return BackendResult(success=True, message="翻译完成。", data={"rows": preview_rows})


def command_load_emotion(args: argparse.Namespace) -> BackendResult:
    from voice_modules.emotion_labeling.emotion_labeling import load_emotion_rows

    return BackendResult(success=True, message="情感标定结果加载完成。", data={"rows": load_emotion_rows(args.role)})


def command_run_emotion(args: argparse.Namespace) -> BackendResult:
    from voice_modules.emotion_labeling.emotion_labeling import run_emotion_labeling

    rows = run_emotion_labeling(
        args.role,
        limit=args.limit if args.limit > 0 else None,
        model=args.model,
        text_model=args.text_model,
        sha256=args.sha256,
        force=args.force,
    )
    return BackendResult(
        success=True,
        message="情感标定完成。",
        output_paths={"workflow_db": str(ensure_workflow_db(args.role).path)},
        data={"rows": rows},
    )


def command_run_emotion_description(args: argparse.Namespace) -> BackendResult:
    from voice_modules.emotion_labeling.emotion_labeling import run_description_only

    rows = run_description_only(
        args.role,
        limit=args.limit if args.limit > 0 else None,
        model=args.model,
        sha256=args.sha256,
        force=args.force,
    )
    return BackendResult(
        success=True,
        message="情感描述完成。",
        output_paths={"workflow_db": str(ensure_workflow_db(args.role).path)},
        data={"rows": rows},
    )


def command_run_emotion_keywords(args: argparse.Namespace) -> BackendResult:
    from voice_modules.emotion_labeling.emotion_labeling import run_keyword_only

    rows = run_keyword_only(
        args.role,
        text_model=args.text_model,
        limit=args.limit if args.limit > 0 else None,
        sha256=args.sha256,
        force=args.force,
    )
    return BackendResult(
        success=True,
        message="关键词提取完成。",
        output_paths={"workflow_db": str(ensure_workflow_db(args.role).path)},
        data={"rows": rows},
    )


def command_save_emotion(args: argparse.Namespace) -> BackendResult:
    from voice_modules.emotion_labeling.emotion_labeling import save_emotion_rows

    rows = read_json(args.input_json)
    if not isinstance(rows, list):
        raise ValueError("情感标定保存数据必须是数组。")
    path = save_emotion_rows(args.role, rows)
    return BackendResult(success=True, message="情感标定已保存。", output_paths={"workflow_db": str(path)}, data={"row_count": len(rows)})


def command_tts(args: argparse.Namespace) -> BackendResult:
    from voice_modules.dubbing.output_overview import list_tts_runs
    from voice_modules.dubbing.tts import run_tts_pipeline

    print("开始执行配音流程...", flush=True)
    selected_results = read_json(args.selected_results_json) if args.selected_results_json else None
    style_results = read_json(args.style_results_json) if args.style_results_json else None
    summary = run_tts_pipeline(
        role=args.role,
        transcript_text=args.transcript_text,
        retrieval_voice_guidance=args.voice_description,
        synthesis_voice_guidance=args.synthesis_guidance,
        backend=args.backend,
        synthesis_language=args.synthesis_language,
        reference_window=args.reference_window,
        reference_min_seconds=args.reference_min_seconds,
        gap_ms=args.gap_ms,
        voxcpm2_diffusion_steps=args.voxcpm2_diffusion_steps,
        voxcpm2_cfg_value=args.voxcpm2_cfg_value,
        voxcpm2_model_path=Path(args.voxcpm2_model_path).resolve(),
        target_model=args.tts_target_model,
        voice_enrollment_model=args.voice_enrollment_model,
        rerank_model=args.rerank_model,
        text_model=args.text_model,
        selected_results=selected_results,
        style_results=style_results,
        play=args.play,
    )
    return BackendResult(
        success=True,
        message="配音生成完成。",
        output_paths={
            "synthesized_audio_path": summary["synthesized_audio_path"],
            "session_dir": summary["session_dir"],
            "reference_audio_path": summary["reference_audio_path"],
            "retrieval_result_path": summary["retrieval_result_path"],
            "run_summary_path": summary["run_summary_path"],
        },
        data={**summary, "rows": list_tts_runs(args.role)},
    )


def command_retrieve(args: argparse.Namespace) -> BackendResult:
    from voice_modules.dubbing.tts import run_retrieval_stage

    print("开始执行检索流程...", flush=True)
    top_k = max(1, int(args.reference_window))
    summary = run_retrieval_stage(
        role=args.role,
        transcript_text=args.transcript_text,
        retrieval_voice_guidance=args.voice_description,
        rerank_model=args.rerank_model,
        text_model=args.text_model,
        top_k=top_k,
    )
    return BackendResult(
        success=True,
        message="检索完成。",
        output_paths={
            "session_dir": summary["session_dir"],
            "retrieval_result_path": summary["retrieval_result_path"],
        },
        data=summary,
    )


def command_synthesize(args: argparse.Namespace) -> BackendResult:
    from voice_modules.dubbing.output_overview import list_tts_runs
    from voice_modules.dubbing.tts import run_synthesis_from_retrieval

    print("开始执行合成流程...", flush=True)
    selected_results = read_json(args.selected_results_json) if args.selected_results_json else None
    style_results = read_json(args.style_results_json) if args.style_results_json else None
    summary = run_synthesis_from_retrieval(
        role=args.role,
        transcript_text=args.transcript_text,
        retrieval_voice_guidance=args.voice_description,
        synthesis_voice_guidance=args.synthesis_guidance,
        backend=args.backend,
        synthesis_language=args.synthesis_language,
        reference_window=args.reference_window,
        reference_min_seconds=args.reference_min_seconds,
        gap_ms=args.gap_ms,
        voxcpm2_diffusion_steps=args.voxcpm2_diffusion_steps,
        voxcpm2_cfg_value=args.voxcpm2_cfg_value,
        voxcpm2_model_path=Path(args.voxcpm2_model_path).resolve(),
        retrieval_result_path=Path(args.retrieval_result_path).resolve(),
        session_dir=Path(args.session_dir).resolve(),
        target_model=args.tts_target_model,
        voice_enrollment_model=args.voice_enrollment_model,
        selected_results=selected_results,
        style_results=style_results,
        play=args.play,
    )
    return BackendResult(
        success=True,
        message="配音合成完成。",
        output_paths={
            "synthesized_audio_path": summary["synthesized_audio_path"],
            "session_dir": summary["session_dir"],
            "reference_audio_path": summary["reference_audio_path"],
            "retrieval_result_path": summary["retrieval_result_path"],
            "run_summary_path": summary["run_summary_path"],
        },
        data={**summary, "rows": list_tts_runs(args.role)},
    )


def command_auto_guidance(args: argparse.Namespace) -> BackendResult:
    from voice_modules.dubbing.auto_guidance import generate_auto_guidance

    guidance = generate_auto_guidance(args.transcript_text, args.text_model)
    return BackendResult(success=True, message="自动指导生成完成。", data={"guidance": guidance})


def command_split_sentences(args: argparse.Namespace) -> BackendResult:
    from voice_modules.long_text.sentence_splitter import split_and_guide

    enable_thinking = not getattr(args, "no_thinking", False)
    segments = split_and_guide(args.transcript_text, args.text_model, enable_thinking=enable_thinking)
    return BackendResult(success=True, message=f"分句完成，共 {len(segments)} 段。", data={"segments": segments})


def command_list_tts_runs(args: argparse.Namespace) -> BackendResult:
    from voice_modules.dubbing.output_overview import cleanup_nonconforming_tts_runs, list_tts_runs

    cleanup_nonconforming_tts_runs(args.role)
    rows = list_tts_runs(args.role)
    return BackendResult(success=True, message="输出音频列表加载完成。", data={"rows": rows})


def command_delete_tts_run(args: argparse.Namespace) -> BackendResult:
    from voice_modules.dubbing.output_overview import delete_tts_run, list_tts_runs

    data = delete_tts_run(args.role, args.run_summary_path)
    data["rows"] = list_tts_runs(args.role)
    return BackendResult(success=True, message="合成语音已删除。", data=data)


def command_cleanup_tts_runs(args: argparse.Namespace) -> BackendResult:
    from voice_modules.dubbing.output_overview import cleanup_nonconforming_tts_runs, list_tts_runs

    data = cleanup_nonconforming_tts_runs(args.role)
    data["rows"] = list_tts_runs(args.role)
    return BackendResult(success=True, message="非规范 tts_runs 目录已清理。", data=data)


def command_normalize_tts_run_filenames(args: argparse.Namespace) -> BackendResult:
    from voice_modules.dubbing.output_overview import list_tts_runs, normalize_tts_run_filenames

    data = normalize_tts_run_filenames(args.role)
    data["rows"] = list_tts_runs(args.role)
    return BackendResult(success=True, message="tts_runs 文件名已统一为新规范。", data=data)


def add_role_command(sub: argparse._SubParsersAction, name: str, handler) -> argparse.ArgumentParser:
    parser = sub.add_parser(name)
    parser.add_argument("--role", required=True)
    parser.set_defaults(func=handler)
    return parser


def build_parser() -> argparse.ArgumentParser:
    config = AppConfig.load()
    parser = argparse.ArgumentParser(description="Voice GUI backend command runner")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("state").set_defaults(func=command_state)
    sub.add_parser("migrate-workspace").set_defaults(func=command_migrate_workspace)
    import_role = add_role_command(sub, "import-role-from-excel", command_import_role_from_excel)
    import_role.add_argument("--audio-dir", required=True)
    import_role.add_argument("--excel", required=True)
    add_role_command(sub, "sync-workflow-db", command_sync_workflow_db)
    add_role_command(sub, "scan-input-audio", command_scan_input_audio)
    add_role_command(sub, "sync-input-audio", command_sync_input_audio)
    add_role_command(sub, "duration-stats", command_duration_stats)
    add_role_command(sub, "load-manifest", command_load_manifest)
    add_role_command(sub, "load-asr", command_load_asr)
    add_role_command(sub, "load-emotion", command_load_emotion)
    add_role_command(sub, "list-tts-runs", command_list_tts_runs)
    add_role_command(sub, "cleanup-tts-runs", command_cleanup_tts_runs)
    add_role_command(sub, "normalize-tts-run-filenames", command_normalize_tts_run_filenames)

    update = add_role_command(sub, "update-filter-status", command_update_filter_status)
    update.add_argument("--status", required=True, choices=["confirmed", "review", "reject", "excluded", "deleted", "unprocessed"])
    update.add_argument("--input-json", required=True)

    reference = add_role_command(sub, "update-reference", command_update_reference)
    reference.add_argument("--selected", action="store_true")
    reference.add_argument("--input-json", required=True)

    save_reference = add_role_command(sub, "save-reference-selection", command_save_reference_selection)
    save_reference.add_argument("--input-json", required=True)

    filter_audio = add_role_command(sub, "filter-audio", command_filter_audio)
    filter_audio.add_argument("--confirm-threshold", type=float, required=True)
    filter_audio.add_argument("--review-threshold", type=float, required=True)

    save_filter = add_role_command(sub, "save-filter-results", command_save_filter_results)
    save_filter.add_argument("--input-json", required=True)

    run_asr_parser = add_role_command(sub, "run-asr", command_run_asr)
    run_asr_parser.add_argument("--limit", type=int, default=0)
    run_asr_parser.add_argument("--api-key", default=config.dashscope_api_key)
    run_asr_parser.add_argument("--model", default=config.asr_model)
    run_asr_parser.add_argument("--language", choices=["zh", "en", "ja"], default="zh")
    run_asr_parser.add_argument("--sha256", default="")
    run_asr_parser.add_argument("--force", action="store_true")

    save_asr = add_role_command(sub, "save-asr", command_save_asr)
    save_asr.add_argument("--input-json", required=True)

    run_translation = add_role_command(sub, "run-translation", command_run_translation)
    run_translation.add_argument("--input-json", required=True)
    run_translation.add_argument("--language", choices=["zh", "en", "ja"], default="zh")
    run_translation.add_argument("--model", default=config.retrieval_text_model)
    run_translation.add_argument("--sha256", default="")
    run_translation.add_argument("--force", action="store_true")
    run_translation.add_argument("--api-key", default=config.dashscope_api_key)
    run_translation.add_argument("--deepseek-api-key", default=config.deepseek_api_key)

    run_emotion = add_role_command(sub, "run-emotion", command_run_emotion)
    run_emotion.add_argument("--limit", type=int, default=0)
    run_emotion.add_argument("--api-key", default=config.dashscope_api_key)
    run_emotion.add_argument("--deepseek-api-key", default=config.deepseek_api_key)
    run_emotion.add_argument("--model", default=config.emotion_model)
    run_emotion.add_argument("--text-model", default=config.retrieval_text_model)
    run_emotion.add_argument("--sha256", default="")
    run_emotion.add_argument("--force", action="store_true")

    run_emotion_desc = add_role_command(sub, "run-emotion-description", command_run_emotion_description)
    run_emotion_desc.add_argument("--limit", type=int, default=0)
    run_emotion_desc.add_argument("--api-key", default=config.dashscope_api_key)
    run_emotion_desc.add_argument("--model", default=config.emotion_model)
    run_emotion_desc.add_argument("--sha256", default="")
    run_emotion_desc.add_argument("--force", action="store_true")

    run_emotion_kw = add_role_command(sub, "run-emotion-keywords", command_run_emotion_keywords)
    run_emotion_kw.add_argument("--limit", type=int, default=0)
    run_emotion_kw.add_argument("--api-key", default=config.dashscope_api_key)
    run_emotion_kw.add_argument("--deepseek-api-key", default=config.deepseek_api_key)
    run_emotion_kw.add_argument("--text-model", default=config.retrieval_text_model)
    run_emotion_kw.add_argument("--sha256", default="")
    run_emotion_kw.add_argument("--force", action="store_true")

    save_emotion = add_role_command(sub, "save-emotion", command_save_emotion)
    save_emotion.add_argument("--input-json", required=True)

    tts = add_role_command(sub, "tts", command_tts)
    tts.add_argument("--transcript-text", required=True)
    tts.add_argument("--voice-description", required=True)
    tts.add_argument("--backend", choices=["api", "voxcpm2_local_basic", "voxcpm2_local_hifi"], required=True)
    tts.add_argument("--synthesis-language", choices=["zh", "en", "ja"], default=config.synthesis_language)
    tts.add_argument("--reference-window", type=int, default=config.reference_window)
    tts.add_argument("--reference-min-seconds", type=float, default=config.reference_min_seconds)
    tts.add_argument("--gap-ms", type=int, default=config.gap_ms)
    tts.add_argument("--voxcpm2-diffusion-steps", type=int, default=config.voxcpm2_diffusion_steps)
    tts.add_argument("--voxcpm2-cfg-value", type=float, default=config.voxcpm2_cfg_value)
    tts.add_argument("--voxcpm2-model-path", default=config.voxcpm2_model_path)
    tts.add_argument("--api-key", default=config.dashscope_api_key)
    tts.add_argument("--deepseek-api-key", default=config.deepseek_api_key)
    tts.add_argument("--rerank-model", default=config.retrieval_rerank_model)
    tts.add_argument("--text-model", default=config.retrieval_text_model)
    tts.add_argument("--voice-enrollment-model", default=config.voice_enrollment_model)
    tts.add_argument("--tts-target-model", default=config.tts_target_model)
    tts.add_argument("--synthesis-guidance", default="")
    tts.add_argument("--selected-results-json", default="")
    tts.add_argument("--style-results-json", default="")
    tts.add_argument("--play", action="store_true")

    retrieve = add_role_command(sub, "retrieve", command_retrieve)
    retrieve.add_argument("--transcript-text", required=True)
    retrieve.add_argument("--voice-description", required=True)
    retrieve.add_argument("--api-key", default=config.dashscope_api_key)
    retrieve.add_argument("--deepseek-api-key", default=config.deepseek_api_key)
    retrieve.add_argument("--rerank-model", default=config.retrieval_rerank_model)
    retrieve.add_argument("--text-model", default=config.retrieval_text_model)
    retrieve.add_argument("--reference-window", type=int, default=10)

    synthesize = add_role_command(sub, "synthesize", command_synthesize)
    synthesize.add_argument("--transcript-text", required=True)
    synthesize.add_argument("--voice-description", required=True)
    synthesize.add_argument("--backend", choices=["api", "voxcpm2_local_basic", "voxcpm2_local_hifi"], required=True)
    synthesize.add_argument("--synthesis-language", choices=["zh", "en", "ja"], default=config.synthesis_language)
    synthesize.add_argument("--reference-window", type=int, default=config.reference_window)
    synthesize.add_argument("--reference-min-seconds", type=float, default=config.reference_min_seconds)
    synthesize.add_argument("--gap-ms", type=int, default=config.gap_ms)
    synthesize.add_argument("--voxcpm2-diffusion-steps", type=int, default=config.voxcpm2_diffusion_steps)
    synthesize.add_argument("--voxcpm2-cfg-value", type=float, default=config.voxcpm2_cfg_value)
    synthesize.add_argument("--voxcpm2-model-path", default=config.voxcpm2_model_path)
    synthesize.add_argument("--retrieval-result-path", required=True)
    synthesize.add_argument("--session-dir", required=True)
    synthesize.add_argument("--api-key", default=config.dashscope_api_key)
    synthesize.add_argument("--deepseek-api-key", default=config.deepseek_api_key)
    synthesize.add_argument("--voice-enrollment-model", default=config.voice_enrollment_model)
    synthesize.add_argument("--tts-target-model", default=config.tts_target_model)
    synthesize.add_argument("--synthesis-guidance", default="")
    synthesize.add_argument("--selected-results-json", default="")
    synthesize.add_argument("--style-results-json", default="")
    synthesize.add_argument("--play", action="store_true")

    auto_guidance = add_role_command(sub, "auto-guidance", command_auto_guidance)
    auto_guidance.add_argument("--transcript-text", required=True)
    auto_guidance.add_argument("--text-model", default=config.retrieval_text_model)
    auto_guidance.add_argument("--api-key", default=config.dashscope_api_key)
    auto_guidance.add_argument("--deepseek-api-key", default=config.deepseek_api_key)

    split_sentences = add_role_command(sub, "split-sentences", command_split_sentences)
    split_sentences.add_argument("--transcript-text", required=True)
    split_sentences.add_argument("--text-model", default=config.retrieval_text_model)
    split_sentences.add_argument("--api-key", default=config.dashscope_api_key)
    split_sentences.add_argument("--deepseek-api-key", default=config.deepseek_api_key)
    split_sentences.add_argument("--no-thinking", action="store_true", default=False)

    delete_tts_run = add_role_command(sub, "delete-tts-run", command_delete_tts_run)
    delete_tts_run.add_argument("--run-summary-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_env(args)
    try:
        result = args.func(args)
    except Exception as exc:
        traceback.print_exc()
        emit_result(BackendResult(success=False, message=str(exc)))
        return 1
    emit_result(result)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
