"""python -m src.cli samples/demo.mp4 --out result.json"""

import argparse
import json
import sys
from pathlib import Path

from .config import settings
from .run import process_video


def main() -> int:
    ap = argparse.ArgumentParser(description="Video -> structured JSON")
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=Path("result.json"))
    ap.add_argument("--srt", type=Path, help="also write subtitles here")
    ap.add_argument("--keep", action="store_true", help="keep the temp workdir")
    args = ap.parse_args()

    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set — copy .env.example to .env", file=sys.stderr)
        return 2
    if not args.video.exists():
        print(f"no such file: {args.video}", file=sys.stderr)
        return 2

    result = process_video(args.video, keep_workdir=args.keep)

    if args.srt:
        args.srt.write_text(result["srt"], encoding="utf-8")
    result.pop("srt")
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    c = result["cost"]
    print(f"{args.video.name}: {c['frames']} frames · {result['elapsed_sec']}s · "
          f"${c['total_usd']} · -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
