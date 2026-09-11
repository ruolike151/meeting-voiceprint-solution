from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from whisperx_stt.audio import get_audio_duration_seconds, iter_audio_files, normalize_audio
from whisperx_stt.transcriber import (
    TranscriptionConfig,
    WhisperXTranscriber,
    write_json,
    write_srt,
    write_txt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark WhisperX on a directory of audio files.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing audio files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--model", default="medium", help="WhisperX model name.")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument("--compute-type", default="auto", help="auto, float16, int8, etc.")
    parser.add_argument("--language", default="auto", help="auto, zh, en, ja, etc.")
    parser.add_argument("--batch-size", default=16, type=int, help="Transcription batch size.")
    parser.add_argument("--no-align", action="store_true", help="Disable WhisperX alignment.")
    parser.add_argument("--no-normalize", action="store_true", help="Skip ffmpeg normalization.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_files = iter_audio_files(args.input_dir)
    if not audio_files:
        raise RuntimeError(f"No supported audio files found in: {args.input_dir}")

    output_dir = args.output_dir.expanduser().resolve()
    transcript_dir = output_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    config = TranscriptionConfig(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        batch_size=args.batch_size,
        align=not args.no_align,
    )
    transcriber = WhisperXTranscriber(config)

    rows: list[dict[str, object]] = []
    for audio_path in audio_files:
        audio_for_model = audio_path
        if not args.no_normalize:
            normalized_path = output_dir / "_normalized_audio" / f"{audio_path.stem}.16k_mono.wav"
            audio_for_model = normalize_audio(audio_path, normalized_path)

        payload = transcriber.transcribe(audio_for_model)
        payload["source_audio_path"] = str(audio_path)

        duration = get_audio_duration_seconds(audio_path)
        elapsed = float(payload["elapsed_seconds"])
        rtf = round(elapsed / duration, 4) if duration else None

        stem = audio_path.stem
        write_json(payload, transcript_dir / f"{stem}.transcript.json")
        write_txt(payload, transcript_dir / f"{stem}.transcript.txt")
        write_srt(payload, transcript_dir / f"{stem}.transcript.srt")

        rows.append(
            {
                "audio_path": str(audio_path),
                "duration_seconds": duration,
                "elapsed_seconds": elapsed,
                "rtf": rtf,
                "model": payload["model"],
                "language": payload["language"],
                "device": payload["device"],
                "compute_type": payload["compute_type"],
                "segment_count": len(payload["segments"]),
            }
        )

    write_benchmark_outputs(rows, output_dir)
    print(f"Processed files: {len(rows)}")
    print(f"Results: {output_dir}")
    return 0


def write_benchmark_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    json_path = output_dir / "benchmark_results.json"
    csv_path = output_dir / "benchmark_results.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "audio_path",
        "duration_seconds",
        "elapsed_seconds",
        "rtf",
        "model",
        "language",
        "device",
        "compute_type",
        "segment_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

