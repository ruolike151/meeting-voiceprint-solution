from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from whisperx_stt.audio import get_audio_duration_seconds, normalize_audio
from whisperx_stt.transcriber import (
    TranscriptionConfig,
    WhisperXTranscriber,
    format_timestamp,
    write_json,
    write_srt,
    write_txt,
)


@dataclass(slots=True)
class SpeakerProfile:
    employee_id: str
    audio_path: Path
    normalized_audio_path: Path
    duration_seconds: float | None
    embedding_backend: str
    embedding_count: int
    centroid: np.ndarray
    embeddings: np.ndarray


@dataclass(slots=True)
class WindowPrediction:
    start: float
    end: float
    speaker: str
    score: float
    margin: float
    confidence: float
    status: str
    scores: dict[str, float]


class MfccEmbeddingExtractor:
    """Lightweight local voiceprint fallback based on MFCC statistics."""

    name = "mfcc"
    description = "MFCC 统计声学特征（轻量兜底，不需要下载预训练声纹模型）"
    model_path: str | None = None
    model_size_mb: float | None = None

    def __init__(self, fallback_reason: str | None = None) -> None:
        self.fallback_reason = fallback_reason

    def extract(self, samples: np.ndarray, sample_rate: int) -> np.ndarray | None:
        return extract_mfcc_embedding(samples, sample_rate)

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "model_path": self.model_path,
            "model_size_mb": self.model_size_mb,
        }
        if self.fallback_reason:
            payload["fallback_reason"] = self.fallback_reason
        return payload


class SpeechBrainEcapaExtractor:
    """Pretrained SpeechBrain ECAPA-TDNN speaker embedding backend."""

    name = "speechbrain_ecapa"
    description = "SpeechBrain ECAPA-TDNN 预训练说话人 embedding（VoxCeleb）"

    def __init__(
        self,
        model_path: Path,
        requested_device: str,
        min_embedding_ckpt_mb: float,
    ) -> None:
        self.model_path = model_path.expanduser().resolve()
        validate_speechbrain_ecapa_dir(self.model_path, min_embedding_ckpt_mb)
        self.device = resolve_torch_device(requested_device)

        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import FetchConfig, LocalStrategy

        runtime_dir = self.model_path / ".runtime_cache"
        self.classifier = EncoderClassifier.from_hparams(
            source=str(self.model_path),
            savedir=str(runtime_dir),
            overrides={"pretrained_path": self.model_path.as_posix()},
            local_strategy=LocalStrategy.NO_LINK,
            fetch_config=FetchConfig(allow_network=False),
            run_opts={"device": self.device},
        )
        self.model_size_mb = directory_size_mb(self.model_path)

    def extract(self, samples: np.ndarray, sample_rate: int) -> np.ndarray | None:
        import torch
        import torchaudio

        if samples.size < int(0.5 * sample_rate):
            return None

        waveform = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        with torch.inference_mode():
            embedding = self.classifier.encode_batch(waveform, normalize=False)
        vector = embedding.squeeze().detach().cpu().numpy().astype(np.float32)
        return l2_normalize(vector)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "model_path": str(self.model_path),
            "model_size_mb": round(self.model_size_mb, 2),
            "device": self.device,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small end-to-end WhisperX + voiceprint demo."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT.parent / "data",
        help="Directory containing meeting.mp3 and A/B/C/D_solo.mp3.",
    )
    parser.add_argument("--meeting", default="meeting.mp3", help="Meeting audio file name.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "voiceprint_demo",
        help="Output directory.",
    )
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "faster-whisper-tiny"),
        help="WhisperX/faster-whisper model name or local path.",
    )
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto.")
    parser.add_argument("--compute-type", default="float16", help="float16, int8, or auto.")
    parser.add_argument("--language", default="auto", help="auto, zh, en, etc.")
    parser.add_argument("--batch-size", default=8, type=int, help="WhisperX batch size.")
    parser.add_argument("--align", action="store_true", help="Enable WhisperX alignment.")
    parser.add_argument(
        "--window-seconds",
        default=2.5,
        type=float,
        help="Speaker matching window length.",
    )
    parser.add_argument(
        "--hop-seconds",
        default=1.25,
        type=float,
        help="Speaker matching hop length.",
    )
    parser.add_argument(
        "--min-score",
        default=-1.0,
        type=float,
        help="Best score below this value is marked as low confidence.",
    )
    parser.add_argument(
        "--min-margin",
        default=0.005,
        type=float,
        help="Best-vs-second score margin below this value is marked as low confidence.",
    )
    parser.add_argument(
        "--reuse-transcript",
        action="store_true",
        help="Reuse output meeting.transcript.json if it already exists.",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=PROJECT_ROOT / "config" / "speaker_corrections.demo.json",
        help="Optional JSON file used to override demo speaker/text results.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "speechbrain_ecapa", "mfcc"],
        default="auto",
        help=(
            "Speaker embedding backend. auto tries SpeechBrain ECAPA first and "
            "falls back to MFCC if the local pretrained model is unavailable."
        ),
    )
    parser.add_argument(
        "--speaker-model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "speechbrain-spkrec-ecapa-voxceleb",
        help="Local SpeechBrain ECAPA model directory.",
    )
    parser.add_argument(
        "--ecapa-min-ckpt-mb",
        default=50.0,
        type=float,
        help="Minimum valid size for SpeechBrain embedding_model.ckpt, in MB.",
    )
    parser.add_argument(
        "--strict-embedding-backend",
        action="store_true",
        help="Fail instead of falling back to MFCC when SpeechBrain ECAPA cannot be loaded.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    normalized_dir = output_dir / "normalized_audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    meeting_path = data_dir / args.meeting
    solo_paths = {
        "A": data_dir / "A_solo.mp3",
        "B": data_dir / "B_solo.mp3",
        "C": data_dir / "C_solo.mp3",
        "D": data_dir / "D_solo.mp3",
    }
    validate_inputs(meeting_path, solo_paths)

    normalized_meeting = normalize_audio(
        meeting_path,
        normalized_dir / "meeting.16k_mono.wav",
    )
    normalized_solos = {
        speaker: normalize_audio(path, normalized_dir / f"{speaker}_solo.16k_mono.wav")
        for speaker, path in solo_paths.items()
    }

    transcript = load_or_transcribe_meeting(args, output_dir, normalized_meeting, meeting_path)
    embedding_extractor = create_embedding_extractor(args)
    print(f"Speaker embedding backend: {embedding_extractor.metadata()['name']}")
    profiles = build_speaker_profiles(normalized_solos, solo_paths, args, embedding_extractor)
    write_profiles(
        profiles,
        output_dir / "voiceprint_profiles.json",
        embedding_extractor.metadata(),
    )

    windows = classify_meeting_windows(
        normalized_meeting,
        profiles,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        min_score=args.min_score,
        min_margin=args.min_margin,
        embedding_extractor=embedding_extractor,
    )
    turns = merge_windows_to_turns(windows, max_gap_seconds=args.hop_seconds)
    transcript_segments = attach_speakers_to_transcript(transcript.get("segments", []), windows)
    corrections = load_corrections(args.corrections)
    if corrections:
        transcript_segments = corrections["segments"]
        turns = corrected_segments_to_turns(transcript_segments)
        full_text = "".join(segment["text"] for segment in transcript_segments)
        correction_source = str(args.corrections.expanduser().resolve())
    else:
        full_text = transcript.get("text", "")
        correction_source = None

    result = {
        "demo_name": "four_person_meeting_voiceprint_demo",
        "data_dir": str(data_dir),
        "meeting_audio": str(meeting_path),
        "normalized_meeting_audio": str(normalized_meeting),
        "solo_audios": {speaker: str(path) for speaker, path in solo_paths.items()},
        "model": args.model,
        "device": transcript.get("device"),
        "compute_type": transcript.get("compute_type"),
        "language": transcript.get("language"),
        "embedding_backend": embedding_extractor.metadata(),
        "meeting_duration_seconds": get_audio_duration_seconds(normalized_meeting),
        "speaker_profiles": {
            profile.employee_id: {
                "source_audio": str(profile.audio_path),
                "normalized_audio": str(profile.normalized_audio_path),
                "duration_seconds": profile.duration_seconds,
                "embedding_backend": profile.embedding_backend,
                "embedding_count": profile.embedding_count,
            }
            for profile in profiles.values()
        },
        "speaker_turns": turns,
        "transcript_segments": transcript_segments,
        "full_text": full_text,
        "correction_source": correction_source,
        "participants": corrections.get("participants") if corrections else list(profiles.keys()),
        "workflow_payload": {
            "meeting_id": "demo_meeting_001",
            "participants": list(profiles.keys()),
            "segments": transcript_segments,
        },
    }

    write_json(result, output_dir / "final_demo_result.json")
    write_corrected_transcript_outputs(result, output_dir)
    write_windows_csv(windows, output_dir / "speaker_windows.csv")
    write_turns_markdown(result, output_dir / "meeting_transcript_with_speakers.md")
    write_summary(result, output_dir / "demo_summary.md")

    print(f"Demo output: {output_dir}")
    print(f"Transcript segments: {len(transcript_segments)}")
    print(f"Speaker windows: {len(windows)}")
    print(f"Speaker turns: {len(turns)}")
    print(f"Final JSON: {output_dir / 'final_demo_result.json'}")
    print(f"Readable transcript: {output_dir / 'meeting_transcript_with_speakers.md'}")
    return 0


def create_embedding_extractor(args: argparse.Namespace) -> MfccEmbeddingExtractor | SpeechBrainEcapaExtractor:
    requested = args.embedding_backend
    if requested in {"auto", "speechbrain_ecapa"}:
        try:
            return SpeechBrainEcapaExtractor(
                model_path=args.speaker_model_path,
                requested_device=args.device,
                min_embedding_ckpt_mb=args.ecapa_min_ckpt_mb,
            )
        except Exception as exc:
            base_message = f"SpeechBrain ECAPA backend is unavailable: {exc}"
            if requested == "speechbrain_ecapa" and args.strict_embedding_backend:
                raise RuntimeError(base_message) from exc
            message = f"{base_message}. Falling back to the local MFCC voiceprint backend."
            print(f"Warning: {message}", file=sys.stderr)
            return MfccEmbeddingExtractor(fallback_reason=str(exc))

    return MfccEmbeddingExtractor()


def validate_speechbrain_ecapa_dir(model_dir: Path, min_embedding_ckpt_mb: float) -> None:
    required_files = [
        "hyperparams.yaml",
        "embedding_model.ckpt",
        "mean_var_norm_emb.ckpt",
        "classifier.ckpt",
        "label_encoder.txt",
    ]
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"model directory is missing required files: {joined}")

    embedding_ckpt = model_dir / "embedding_model.ckpt"
    embedding_mb = embedding_ckpt.stat().st_size / (1024 * 1024)
    if embedding_mb < min_embedding_ckpt_mb:
        raise RuntimeError(
            f"{embedding_ckpt.name} is only {embedding_mb:.2f} MB; "
            "the complete SpeechBrain ECAPA weight file is expected to be about 80 MB"
        )


def directory_size_mb(path: Path) -> float:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / (1024 * 1024)


def resolve_torch_device(requested_device: str) -> str:
    import torch

    requested = requested_device.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested_device


def validate_inputs(meeting_path: Path, solo_paths: dict[str, Path]) -> None:
    missing = [path for path in [meeting_path, *solo_paths.values()] if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required demo audio files:\n{joined}")


def load_corrections(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"Correction file has no segments: {resolved}")
    payload["segments"] = [
        {
            "id": int(segment.get("id", index)),
            "start": round(float(segment["start"]), 3),
            "end": round(float(segment["end"]), 3),
            "speaker": str(segment["speaker"]),
            "speaker_confidence": float(segment.get("speaker_confidence", 1.0)),
            "speaker_status": str(segment.get("speaker_status", "corrected")),
            "text": str(segment["text"]),
        }
        for index, segment in enumerate(segments)
    ]
    return payload


def corrected_segments_to_turns(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start": segment["start"],
            "end": segment["end"],
            "speaker": segment["speaker"],
            "status": segment.get("speaker_status", "corrected"),
            "avg_score": 1.0,
            "avg_confidence": segment.get("speaker_confidence", 1.0),
        }
        for segment in segments
    ]


def load_or_transcribe_meeting(
    args: argparse.Namespace,
    output_dir: Path,
    normalized_meeting: Path,
    source_meeting: Path,
) -> dict[str, Any]:
    transcript_path = output_dir / "meeting.transcript.json"
    if args.reuse_transcript and transcript_path.exists():
        return json.loads(transcript_path.read_text(encoding="utf-8"))

    config = TranscriptionConfig(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        batch_size=args.batch_size,
        align=args.align,
    )
    transcriber = WhisperXTranscriber(config)
    transcript = transcriber.transcribe(normalized_meeting)
    transcript["source_audio_path"] = str(source_meeting)

    write_json(transcript, transcript_path)
    write_txt(transcript, output_dir / "meeting.transcript.txt")
    write_srt(transcript, output_dir / "meeting.transcript.srt")
    return transcript


def build_speaker_profiles(
    normalized_solos: dict[str, Path],
    solo_paths: dict[str, Path],
    args: argparse.Namespace,
    embedding_extractor: MfccEmbeddingExtractor | SpeechBrainEcapaExtractor,
) -> dict[str, SpeakerProfile]:
    profiles: dict[str, SpeakerProfile] = {}
    for speaker, normalized_path in normalized_solos.items():
        samples, sample_rate = read_wav_mono(normalized_path)
        embeddings = extract_window_embeddings(
            samples,
            sample_rate,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            embedding_extractor=embedding_extractor,
        )
        if not embeddings:
            embedding = embedding_extractor.extract(samples, sample_rate)
            if embedding is None:
                raise RuntimeError(f"No valid speech embedding extracted for {speaker}.")
            embeddings = [embedding]

        matrix = np.stack(embeddings)
        centroid = l2_normalize(matrix.mean(axis=0))
        profiles[speaker] = SpeakerProfile(
            employee_id=speaker,
            audio_path=solo_paths[speaker],
            normalized_audio_path=normalized_path,
            duration_seconds=get_audio_duration_seconds(normalized_path),
            embedding_backend=embedding_extractor.metadata()["name"],
            embedding_count=len(embeddings),
            centroid=centroid,
            embeddings=matrix,
        )
    return profiles


def classify_meeting_windows(
    normalized_meeting: Path,
    profiles: dict[str, SpeakerProfile],
    window_seconds: float,
    hop_seconds: float,
    min_score: float,
    min_margin: float,
    embedding_extractor: MfccEmbeddingExtractor | SpeechBrainEcapaExtractor,
) -> list[WindowPrediction]:
    samples, sample_rate = read_wav_mono(normalized_meeting)
    total_seconds = len(samples) / sample_rate
    predictions: list[WindowPrediction] = []
    start = 0.0
    while start < total_seconds:
        end = min(start + window_seconds, total_seconds)
        if end - start < 0.8:
            break
        segment = slice_samples(samples, sample_rate, start, end)
        embedding = embedding_extractor.extract(segment, sample_rate)
        if embedding is not None:
            prediction = classify_embedding(
                embedding,
                profiles,
                start=start,
                end=end,
                min_score=min_score,
                min_margin=min_margin,
            )
            predictions.append(prediction)
        start += hop_seconds
    return predictions


def classify_embedding(
    embedding: np.ndarray,
    profiles: dict[str, SpeakerProfile],
    start: float,
    end: float,
    min_score: float,
    min_margin: float,
) -> WindowPrediction:
    scores: dict[str, float] = {}
    for speaker, profile in profiles.items():
        similarities = profile.embeddings @ embedding
        top_k = np.sort(similarities)[-min(3, len(similarities)) :]
        scores[speaker] = float(top_k.mean())

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_speaker, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = best_score - second_score
    confidence = score_to_confidence(best_score, margin)
    status = "accepted" if best_score >= min_score and margin >= min_margin else "needs_review"
    return WindowPrediction(
        start=round(start, 3),
        end=round(end, 3),
        speaker=best_speaker,
        score=round(best_score, 4),
        margin=round(margin, 4),
        confidence=round(confidence, 4),
        status=status,
        scores={speaker: round(score, 4) for speaker, score in scores.items()},
    )


def score_to_confidence(score: float, margin: float) -> float:
    score_part = (score + 1.0) / 2.0
    margin_part = 1.0 / (1.0 + math.exp(-18.0 * margin))
    return max(0.0, min(1.0, 0.65 * score_part + 0.35 * margin_part))


def merge_windows_to_turns(
    windows: list[WindowPrediction],
    max_gap_seconds: float,
) -> list[dict[str, Any]]:
    if not windows:
        return []

    turns: list[dict[str, Any]] = []
    current = {
        "start": windows[0].start,
        "end": windows[0].end,
        "speaker": windows[0].speaker,
        "status": windows[0].status,
        "scores": [windows[0].score],
        "confidences": [windows[0].confidence],
        "statuses": [windows[0].status],
    }
    for window in windows[1:]:
        same_speaker = window.speaker == current["speaker"]
        close_enough = window.start <= float(current["end"]) + max_gap_seconds
        if same_speaker and close_enough:
            current["end"] = window.end
            current["scores"].append(window.score)
            current["confidences"].append(window.confidence)
            current["statuses"].append(window.status)
        else:
            turns.append(finalize_turn(current))
            current = {
                "start": window.start,
                "end": window.end,
                "speaker": window.speaker,
                "status": window.status,
                "scores": [window.score],
                "confidences": [window.confidence],
                "statuses": [window.status],
            }
    turns.append(finalize_turn(current))
    return turns


def finalize_turn(turn: dict[str, Any]) -> dict[str, Any]:
    scores = turn.pop("scores")
    confidences = turn.pop("confidences")
    statuses = turn.pop("statuses")
    turn["start"] = round(float(turn["start"]), 3)
    turn["end"] = round(float(turn["end"]), 3)
    turn["avg_score"] = round(float(np.mean(scores)), 4)
    turn["avg_confidence"] = round(float(np.mean(confidences)), 4)
    accepted_ratio = statuses.count("accepted") / max(1, len(statuses))
    turn["status"] = "accepted" if accepted_ratio >= 0.5 else "needs_review"
    return turn


def attach_speakers_to_transcript(
    segments: list[dict[str, Any]],
    windows: list[WindowPrediction],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        speaker, confidence, status = vote_speaker_for_span(start, end, windows)
        enriched.append(
            {
                "id": segment.get("id"),
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": speaker,
                "speaker_confidence": confidence,
                "speaker_status": status,
                "text": segment.get("text", ""),
            }
        )
    return enriched


def vote_speaker_for_span(
    start: float,
    end: float,
    windows: list[WindowPrediction],
) -> tuple[str, float, str]:
    votes: dict[str, float] = {}
    confidences: dict[str, list[float]] = {}
    statuses: dict[str, list[str]] = {}
    for window in windows:
        overlap = max(0.0, min(end, window.end) - max(start, window.start))
        if overlap <= 0:
            continue
        weight = overlap * max(0.01, window.confidence)
        votes[window.speaker] = votes.get(window.speaker, 0.0) + weight
        confidences.setdefault(window.speaker, []).append(window.confidence)
        statuses.setdefault(window.speaker, []).append(window.status)

    if not votes:
        return "UNKNOWN", 0.0, "needs_review"

    speaker = max(votes.items(), key=lambda item: item[1])[0]
    confidence = round(float(np.mean(confidences[speaker])), 4)
    accepted_ratio = statuses[speaker].count("accepted") / max(1, len(statuses[speaker]))
    status = "accepted" if accepted_ratio >= 0.5 else "needs_review"
    return speaker, confidence, status


def extract_window_embeddings(
    samples: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
    embedding_extractor: MfccEmbeddingExtractor | SpeechBrainEcapaExtractor,
) -> list[np.ndarray]:
    total_seconds = len(samples) / sample_rate
    embeddings: list[np.ndarray] = []
    start = 0.0
    while start < total_seconds:
        end = min(start + window_seconds, total_seconds)
        if end - start < 0.8:
            break
        embedding = embedding_extractor.extract(slice_samples(samples, sample_rate, start, end), sample_rate)
        if embedding is not None:
            embeddings.append(embedding)
        start += hop_seconds
    return embeddings


def extract_mfcc_embedding(samples: np.ndarray, sample_rate: int) -> np.ndarray | None:
    import torch
    import torchaudio

    if samples.size < int(0.5 * sample_rate):
        return None

    waveform = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)
    n_fft = 512

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=30,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": 48,
            "center": False,
            "power": 2.0,
        },
    )
    with torch.no_grad():
        mfcc = mfcc_transform(waveform).squeeze(0).transpose(0, 1).cpu().numpy()

    rms = frame_rms(samples, frame_length=frame_length, hop_length=hop_length)
    usable = min(len(mfcc), len(rms))
    if usable < 5:
        return None
    mfcc = mfcc[:usable]
    rms = rms[:usable]

    threshold = max(float(np.quantile(rms, 0.55)) * 0.65, float(np.max(rms)) * 0.035, 1e-5)
    voiced = rms > threshold
    if int(voiced.sum()) < 5:
        voiced = rms > max(float(np.max(rms)) * 0.02, 1e-6)
    if int(voiced.sum()) < 5:
        return None

    voiced_mfcc = mfcc[voiced]
    voiced_rms = rms[voiced]
    cepstra = voiced_mfcc[:, 1:25]
    delta = np.diff(cepstra, axis=0) if len(cepstra) > 1 else np.zeros_like(cepstra)

    features = np.concatenate(
        [
            cepstra.mean(axis=0),
            cepstra.std(axis=0),
            delta.mean(axis=0),
            delta.std(axis=0),
            np.array(
                [
                    voiced_rms.mean(),
                    voiced_rms.std(),
                    np.percentile(voiced_rms, 25),
                    np.percentile(voiced_rms, 75),
                ],
                dtype=np.float32,
            ),
        ]
    )
    return l2_normalize(features.astype(np.float32))


def frame_rms(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(samples) < frame_length:
        return np.array([float(np.sqrt(np.mean(samples**2) + 1e-10))], dtype=np.float32)
    values = []
    for start in range(0, len(samples) - frame_length + 1, hop_length):
        frame = samples[start : start + frame_length]
        values.append(float(np.sqrt(np.mean(frame**2) + 1e-10)))
    return np.asarray(values, dtype=np.float32)


def slice_samples(samples: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    start_index = max(0, int(round(start * sample_rate)))
    end_index = min(len(samples), int(round(end * sample_rate)))
    return samples[start_index:end_index]


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak > 0:
        mono = mono / max(peak, 1.0)
    return mono.astype(np.float32), int(sample_rate)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def write_profiles(
    profiles: dict[str, SpeakerProfile],
    output_path: Path,
    embedding_backend: dict[str, Any],
) -> None:
    payload = {
        "embedding_backend": embedding_backend,
        "profiles": {
            speaker: {
                "source_audio": str(profile.audio_path),
                "normalized_audio": str(profile.normalized_audio_path),
                "duration_seconds": profile.duration_seconds,
                "embedding_backend": profile.embedding_backend,
                "embedding_count": profile.embedding_count,
                "centroid_dimension": int(profile.centroid.shape[0]),
            }
            for speaker, profile in profiles.items()
        },
    }
    write_json(payload, output_path)


def write_corrected_transcript_outputs(result: dict[str, Any], output_dir: Path) -> None:
    payload = {
        "audio_path": result["normalized_meeting_audio"],
        "source_audio_path": result["meeting_audio"],
        "language": result["language"],
        "correction_source": result.get("correction_source"),
        "segments": [
            {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "speaker": segment["speaker"],
                "text": segment["text"],
            }
            for segment in result["transcript_segments"]
        ],
        "text": result["full_text"],
    }
    write_json(payload, output_dir / "meeting.corrected_transcript.json")
    write_txt(payload, output_dir / "meeting.corrected_transcript.txt")
    write_srt(payload, output_dir / "meeting.corrected_transcript.srt")


def write_windows_csv(windows: list[WindowPrediction], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "start",
        "end",
        "speaker",
        "score",
        "margin",
        "confidence",
        "status",
        "A",
        "B",
        "C",
        "D",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for window in windows:
            row: dict[str, Any] = {
                "start": window.start,
                "end": window.end,
                "speaker": window.speaker,
                "score": window.score,
                "margin": window.margin,
                "confidence": window.confidence,
                "status": window.status,
            }
            row.update(window.scores)
            writer.writerow(row)


def write_turns_markdown(result: dict[str, Any], output_path: Path) -> None:
    participants = result.get("participants", {})
    lines = [
        "# 四人会议 Demo 转写与身份标注",
        "",
        f"- 会议音频：`{result['meeting_audio']}`",
        f"- 模型：`{result['model']}`",
        f"- 声纹后端：`{result.get('embedding_backend', {}).get('name', 'unknown')}`",
        f"- 设备：`{result['device']}`",
        f"- 语言：`{result['language']}`",
    ]
    if result.get("correction_source"):
        lines.append(f"- 校正文件：`{result['correction_source']}`")
    lines.extend(
        [
            "",
            "## 转写片段",
            "",
            "| 时间 | 说话人 | 置信度 | 状态 | 文本 |",
            "|---|---|---:|---|---|",
        ]
    )
    for segment in result["transcript_segments"]:
        time_range = f"{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}"
        speaker = format_speaker(segment["speaker"], participants)
        status = format_status(segment["speaker_status"])
        text = str(segment["text"]).replace("|", "\\|")
        lines.append(
            f"| {time_range} | {speaker} | "
            f"{segment['speaker_confidence']:.3f} | {status} | {text} |"
        )

    lines.extend(["", "## 声纹时间轴", "", "| 时间 | 说话人 | 平均置信度 | 状态 |"])
    lines.append("|---|---|---:|---|")
    for turn in result["speaker_turns"]:
        time_range = f"{format_timestamp(turn['start'])} - {format_timestamp(turn['end'])}"
        speaker = format_speaker(turn["speaker"], participants)
        status = format_status(turn["status"])
        lines.append(
            f"| {time_range} | {speaker} | "
            f"{turn['avg_confidence']:.3f} | {status} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(result: dict[str, Any], output_path: Path) -> None:
    participants = result.get("participants", {})
    speaker_seconds: dict[str, float] = {}
    for turn in result["speaker_turns"]:
        duration = float(turn["end"]) - float(turn["start"])
        speaker_seconds[turn["speaker"]] = speaker_seconds.get(turn["speaker"], 0.0) + duration

    lines = [
        "# 四人会议声纹 Demo 摘要",
        "",
        "## 输入",
        "",
        f"- 会议录音：`{result['meeting_audio']}`",
        f"- 单人注册录音：`{', '.join(result['solo_audios'].keys())}`",
        f"- 会议时长：`{result['meeting_duration_seconds']}` 秒",
        f"- 声纹后端：`{result.get('embedding_backend', {}).get('name', 'unknown')}`",
    ]
    backend = result.get("embedding_backend", {})
    if backend.get("model_path"):
        lines.append(f"- 声纹模型路径：`{backend['model_path']}`")
    if backend.get("fallback_reason"):
        lines.append(f"- 声纹后端回退原因：{backend['fallback_reason']}")
    if result.get("correction_source"):
        lines.append(f"- 人物与文本校正：已使用 `{result['correction_source']}`")
    if isinstance(participants, dict):
        lines.extend(["", "## 人物对应关系", "", "| 人物 | 角色 |", "|---|---|"])
        for speaker, role in participants.items():
            lines.append(f"| {speaker} | {role} |")
    lines.extend(
        [
            "",
            "## 校正后发言时长",
            "",
            "| 说话人 | 发言时长（秒） |",
            "|---|---:|",
        ]
    )
    for speaker, seconds in sorted(speaker_seconds.items()):
        lines.append(f"| {format_speaker(speaker, participants)} | {seconds:.2f} |")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "本 demo 保留了自动声纹匹配结果，同时对本次会议的人物和文本输出应用了人工校正文件。",
            "这样既能展示完整工程流程，也能避免演示文档里出现明显的人物归属错误。",
            "推荐生产路线是使用 ECAPA-TDNN/x-vector/pyannote 等预训练 speaker embedding 构建员工声纹库；当本地预训练模型缺失时，本脚本会回退到 MFCC 兜底后端，保证 demo 可运行。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_speaker(speaker: str, participants: Any) -> str:
    if isinstance(participants, dict) and speaker in participants:
        return f"{speaker}（{participants[speaker]}）"
    return speaker


def format_status(status: str) -> str:
    return {
        "accepted": "自动通过",
        "needs_review": "需复核",
        "corrected": "已校正",
    }.get(status, status)


if __name__ == "__main__":
    raise SystemExit(main())
