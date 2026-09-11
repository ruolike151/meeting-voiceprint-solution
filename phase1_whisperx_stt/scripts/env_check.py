from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from whisperx_stt.audio import ensure_ffmpeg_available_for_child_processes
from whisperx_stt.transcriber import configure_local_caches


def main() -> int:
    cache_root = configure_local_caches()

    import imageio_ffmpeg
    import torch
    import torchaudio
    import torchvision
    import whisperx

    print(f"python: {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")
    print(f"project_root: {PROJECT_ROOT}")
    print(f"cache_root: {cache_root}")
    print(f"torch: {torch.__version__}")
    print(f"torchaudio: {torchaudio.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"whisperx: {getattr(whisperx, '__version__', 'installed')}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"imageio_ffmpeg: {imageio_ffmpeg.get_ffmpeg_exe()}")
    print(f"ffmpeg_for_whisperx: {ensure_ffmpeg_available_for_child_processes()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

