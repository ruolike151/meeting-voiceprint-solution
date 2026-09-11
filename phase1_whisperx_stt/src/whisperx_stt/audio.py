from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


SUPPORTED_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


def ensure_audio_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Expected an audio file, got directory: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError(
            f"Unsupported audio suffix '{resolved.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_SUFFIXES))}"
        )
    return resolved


def require_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary:
        return binary
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError(
                "ffmpeg was not found in PATH and imageio-ffmpeg is not installed. "
                "Run: pip install imageio-ffmpeg"
            ) from exc
    raise RuntimeError(f"{name} was not found in PATH.")


def ensure_ffmpeg_available_for_child_processes() -> str:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary

    ffmpeg = require_binary("ffmpeg")
    wrapper_dir = Path(__file__).resolve().parents[2] / ".bin"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = wrapper_dir / "ffmpeg.exe"
    if not wrapper_path.exists():
        shutil.copy2(ffmpeg, wrapper_path)
    os.environ["PATH"] = f"{wrapper_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    return str(wrapper_path)


def normalize_audio(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    source = ensure_audio_file(input_path)
    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_binary("ffmpeg")

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-vn",
        str(output),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    return output


def get_audio_duration_seconds(path: Path) -> float | None:
    source = ensure_audio_file(path)
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(source),
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            try:
                payload: dict[str, Any] = json.loads(completed.stdout)
                duration = payload.get("format", {}).get("duration")
                return float(duration) if duration is not None else None
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

    try:
        import soundfile as sf

        info = sf.info(str(source))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def iter_audio_files(input_dir: Path) -> list[Path]:
    root = input_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected input directory, got file: {root}")

    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    ]
    return sorted(files)
