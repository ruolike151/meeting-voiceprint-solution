from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .audio import ensure_ffmpeg_available_for_child_processes


@dataclass(slots=True)
class TranscriptionConfig:
    model: str = "medium"
    device: str = "auto"
    compute_type: str = "auto"
    language: str = "auto"
    batch_size: int = 16
    align: bool = True


class WhisperXTranscriber:
    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config
        configure_local_caches()
        self.device = self._resolve_device(config.device)
        self.compute_type = self._resolve_compute_type(config.compute_type, self.device)

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        configure_local_caches()
        ensure_ffmpeg_available_for_child_processes()
        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError(
                "WhisperX is not installed. Run: pip install -r requirements.txt"
            ) from exc

        started_at = time.perf_counter()
        audio = whisperx.load_audio(str(audio_path))
        model = whisperx.load_model(
            self.config.model,
            self.device,
            compute_type=self.compute_type,
        )

        transcribe_kwargs: dict[str, Any] = {"batch_size": self.config.batch_size}
        if self.config.language != "auto":
            transcribe_kwargs["language"] = self.config.language

        try:
            result = model.transcribe(audio, **transcribe_kwargs)
        except TypeError:
            transcribe_kwargs.pop("language", None)
            result = model.transcribe(audio, **transcribe_kwargs)

        alignment_error: str | None = None
        if self.config.align and result.get("segments"):
            language_code = result.get("language")
            if language_code:
                try:
                    align_model, metadata = whisperx.load_align_model(
                        language_code=language_code,
                        device=self.device,
                    )
                    result = whisperx.align(
                        result["segments"],
                        align_model,
                        metadata,
                        audio,
                        self.device,
                        return_char_alignments=False,
                    )
                    result["language"] = language_code
                except Exception as exc:  # pragma: no cover - depends on model assets.
                    alignment_error = str(exc)

        elapsed_seconds = time.perf_counter() - started_at
        segments = normalize_segments(result.get("segments", []))
        text = "".join(segment["text"] for segment in segments).strip()

        payload: dict[str, Any] = {
            "audio_path": str(audio_path),
            "model": self.config.model,
            "language": result.get("language", self.config.language),
            "device": self.device,
            "compute_type": self.compute_type,
            "batch_size": self.config.batch_size,
            "align": self.config.align,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "segments": segments,
            "text": text,
        }
        if alignment_error:
            payload["alignment_error"] = alignment_error
        return payload

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @staticmethod
    def _resolve_compute_type(compute_type: str, device: str) -> str:
        if compute_type != "auto":
            return compute_type
        return "float16" if device == "cuda" else "int8"


def normalize_segments(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for index, segment in enumerate(raw_segments):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        normalized: dict[str, Any] = {
            "id": index,
            "start": round(float(segment.get("start", 0.0)), 3),
            "end": round(float(segment.get("end", 0.0)), 3),
            "text": text,
        }
        if "words" in segment:
            normalized["words"] = segment["words"]
        segments.append(normalized)
    return segments


def write_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def write_txt(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}] "
        f"{segment['text']}"
        for segment in payload.get("segments", [])
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_srt(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, segment in enumerate(payload.get("segments", []), start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(segment['start'])} --> "
                    f"{format_srt_timestamp(segment['end'])}",
                    segment["text"],
                ]
            )
        )
    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return output_path


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{sec:02d}.{ms:03d}"


def format_srt_timestamp(seconds: float) -> str:
    return format_timestamp(seconds).replace(".", ",")


def config_to_dict(config: TranscriptionConfig) -> dict[str, Any]:
    return asdict(config)


def configure_local_caches() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    cache_root = project_root / ".cache"
    cache_paths = {
        "XDG_CACHE_HOME": cache_root,
        "HF_HOME": cache_root / "huggingface",
        "TORCH_HOME": cache_root / "torch",
        "MPLCONFIGDIR": cache_root / "matplotlib",
    }
    for key, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return cache_root
