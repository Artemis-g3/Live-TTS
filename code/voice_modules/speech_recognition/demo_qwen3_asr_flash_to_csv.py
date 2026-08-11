import argparse
import base64
import csv
import os
import re
import time
from pathlib import Path
from typing import Iterable, List

import dashscope
from dashscope import MultiModalConversation


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
DEFAULT_FIELDS = [
    "role",
    "file_name",
    "file_path",
    "model",
    "language",
    "transcript",
    "error",
]
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Recognize confirmed role audio with qwen3-asr-flash and save results to a CSV."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root / "筛选音频",
        help="Root directory containing per-role confirmed folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root / "语音识别" / "筛选音频识别结果.csv",
        help="CSV file path for ASR results.",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        default=["尤诺", "莫宁"],
        help="Role names to process.",
    )
    parser.add_argument(
        "--model",
        default="qwen3-asr-flash",
        help="Bailian ASR model name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of audio files per role for debugging.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between successful requests.",
    )
    parser.add_argument(
        "--retry-csv",
        type=Path,
        default=None,
        help="Retry only failed rows from an existing CSV, then overwrite that CSV.",
    )
    parser.add_argument(
        "--retry-max-attempts",
        type=int,
        default=4,
        help="Maximum attempts per audio request.",
    )
    parser.add_argument(
        "--retry-base-sleep",
        type=float,
        default=2.0,
        help="Base backoff seconds for retryable failures.",
    )
    return parser.parse_args()


def ensure_api_key() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY.")
    os.environ["DASHSCOPE_API_KEY"] = api_key
    dashscope.api_key = api_key


def iter_audio_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def infer_mime_type(audio_path: Path) -> str:
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }.get(audio_path.suffix.lower(), "application/octet-stream")


def audio_to_data_uri(audio_path: Path) -> str:
    mime_type = infer_mime_type(audio_path)
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def extract_text(response) -> str:
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    if not output:
        return ""

    choices = getattr(output, "choices", None)
    if choices is None and isinstance(output, dict):
        choices = output.get("choices", [])
    if not choices:
        return ""

    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
    if not message:
        return ""

    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if not content:
        return ""

    texts: List[str] = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text" and item.get("text"):
                texts.append(str(item["text"]).strip())
            elif item.get("text"):
                texts.append(str(item["text"]).strip())
    return "\n".join(text for text in texts if text)


def response_error(response) -> str:
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


def parse_status_code(error_text: str) -> int | None:
    match = re.search(r"status_code=(\d+)", error_text or "")
    if match:
        return int(match.group(1))
    return None


def is_retryable_error(error_text: str) -> bool:
    status_code = parse_status_code(error_text)
    return status_code in RETRYABLE_STATUS_CODES


def recognize_once(audio_path: Path, model: str) -> tuple[str, str]:
    data_uri = audio_to_data_uri(audio_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "audio": data_uri,
                }
            ],
        }
    ]

    response = MultiModalConversation.call(
        model=model,
        messages=messages,
        result_format="message",
        asr_options={"language": "zh"},
    )
    error = response_error(response)
    if error:
        return "", error
    return extract_text(response), ""


def recognize_with_retry(
    audio_path: Path,
    model: str,
    max_attempts: int,
    base_sleep: float,
) -> tuple[str, str]:
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        text, error = recognize_once(audio_path, model)
        if not error:
            return text, ""
        last_error = error
        if attempt >= max_attempts or not is_retryable_error(error):
            break
        sleep_seconds = base_sleep * (2 ** (attempt - 1))
        print(
            f"[retry] {audio_path.name} attempt {attempt}/{max_attempts} failed, "
            f"sleep {sleep_seconds:.1f}s: {error}"
        )
        time.sleep(sleep_seconds)
    return "", last_error


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_full(args: argparse.Namespace) -> None:
    rows: List[dict] = []
    total = 0

    for role in args.roles:
        confirmed_dir = args.input_root / role / "confirmed"
        if not confirmed_dir.exists():
            print(f"[skip] missing confirmed dir: {confirmed_dir}")
            continue

        files = list(iter_audio_files(confirmed_dir))
        if args.limit is not None:
            files = files[: args.limit]

        print(f"[{role}] start {len(files)} files")
        for index, audio_path in enumerate(files, start=1):
            text, error = recognize_with_retry(
                audio_path=audio_path,
                model=args.model,
                max_attempts=args.retry_max_attempts,
                base_sleep=args.retry_base_sleep,
            )
            rows.append(
                {
                    "role": role,
                    "file_name": audio_path.name,
                    "file_path": str(audio_path.resolve()),
                    "model": args.model,
                    "language": "zh",
                    "transcript": text,
                    "error": error,
                }
            )
            total += 1

            if index % 20 == 0 or index == len(files):
                print(f"[{role}] processed {index}/{len(files)}")
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    write_csv(args.output_csv, rows)
    print(f"done: {total} rows -> {args.output_csv}")


def run_retry(args: argparse.Namespace) -> None:
    rows = list(csv.DictReader(args.retry_csv.open(encoding="utf-8-sig")))
    retry_indexes = [i for i, row in enumerate(rows) if row.get("error", "").strip()]
    if args.limit is not None:
        retry_indexes = retry_indexes[: args.limit]

    print(f"[retry-csv] start {len(retry_indexes)} failed rows from {args.retry_csv}")
    fixed = 0

    for pos, index in enumerate(retry_indexes, start=1):
        row = rows[index]
        audio_path = Path(row["file_path"])
        text, error = recognize_with_retry(
            audio_path=audio_path,
            model=row.get("model") or args.model,
            max_attempts=args.retry_max_attempts,
            base_sleep=args.retry_base_sleep,
        )
        row["transcript"] = text
        row["error"] = error
        row["language"] = "zh"
        row["model"] = row.get("model") or args.model
        rows[index] = row
        if not error:
            fixed += 1

        if pos % 10 == 0 or pos == len(retry_indexes):
            print(f"[retry-csv] processed {pos}/{len(retry_indexes)}")
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    write_csv(args.retry_csv, rows)
    remaining = sum(1 for row in rows if row.get("error", "").strip())
    print(
        f"[retry-csv] fixed={fixed}, remaining_errors={remaining} -> {args.retry_csv}"
    )


def main() -> None:
    args = parse_args()
    ensure_api_key()

    if args.retry_csv is not None:
        run_retry(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
