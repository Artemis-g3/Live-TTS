"""Download the CAM++ speaker-verification model from ModelScope.

The model directory is not shipped with the repository; run this script to
fetch it into the location expected by the application.

Usage:
    python scripts\\download_campplus.py
    python scripts\\download_campplus.py --target <custom-dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODEL_ID = "damo/speech_campplus_sv_zh-cn_16k-common"
MODEL_REVISION = "v1.0.0"
DEFAULT_TARGET = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "voice_modules"
    / "audio_filter"
    / "speech_campplus_sv_zh-cn_16k-common"
)
EXPECTED_FILES = ("campplus_cn_common.bin", "configuration.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 CAM++ 说话人验证模型（ModelScope）")
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="模型保存目录（默认：项目内 audio_filter 目录）",
    )
    parser.add_argument(
        "--revision",
        default=MODEL_REVISION,
        help=f"ModelScope 模型版本（默认 {MODEL_REVISION}）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = args.target.expanduser().resolve()
    print(f"[Download] 模型: {MODEL_ID}@{args.revision}")
    print(f"[Download] 保存到: {target}")
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print(
            "[Download] 未找到 modelscope，请先安装依赖: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1
    snapshot_download(MODEL_ID, revision=args.revision, local_dir=str(target))
    missing = [name for name in EXPECTED_FILES if not (target / name).exists()]
    if missing:
        print(f"[Download] 警告：以下文件未找到: {', '.join(missing)}", file=sys.stderr)
        return 1
    print(f"[Download] 完成。模型文件已就绪: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
