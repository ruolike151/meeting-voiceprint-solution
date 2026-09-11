from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from whisperx_stt.audio import ensure_audio_file, normalize_audio
from whisperx_stt.transcriber import (
    TranscriptionConfig,
    WhisperXTranscriber,
    write_json,
    write_srt,
    write_txt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe one audio file with WhisperX.")
    parser.add_argument("--input", required=True, type=Path, help="Input audio path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--model", default="medium", help="WhisperX model name.")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument("--compute-type", default="auto", help="auto, float16, int8, etc.")
    parser.add_argument("--language", default="auto", help="auto, zh, en, ja, etc.")
    parser.add_argument("--batch-size", default=16, type=int, help="Transcription batch size.")
    parser.add_argument("--no-align", action="store_true", help="Disable WhisperX alignment.")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip ffmpeg normalization and pass the input directly to WhisperX.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = ensure_audio_file(args.input)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_for_model = input_path
    if not args.no_normalize:
        normalized_dir = output_dir / "_normalized_audio"
        audio_for_model = normalized_dir / f"{input_path.stem}.16k_mono.wav"
        normalize_audio(input_path, audio_for_model)

    config = TranscriptionConfig(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        batch_size=args.batch_size,
        align=not args.no_align,
    )
    transcriber = WhisperXTranscriber(config)
    payload = transcriber.transcribe(audio_for_model)
    payload["source_audio_path"] = str(input_path)

    stem = input_path.stem
    json_path = output_dir / f"{stem}.transcript.json"
    txt_path = output_dir / f"{stem}.transcript.txt"
    srt_path = output_dir / f"{stem}.transcript.srt"
    write_json(payload, json_path)
    write_txt(payload, txt_path)
    write_srt(payload, srt_path)

    print(f"JSON: {json_path}")
    print(f"TXT:  {txt_path}")
    print(f"SRT:  {srt_path}")
    print(f"Elapsed seconds: {payload['elapsed_seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

