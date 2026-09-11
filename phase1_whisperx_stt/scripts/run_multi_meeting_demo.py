from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the voiceprint demo on multiple meeting audio files and write one combined final output."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT.parent / "data",
        help="Directory containing meeting audio files and A/B/C/D_solo.mp3.",
    )
    parser.add_argument(
        "--meetings",
        nargs="+",
        default=["meeting.mp3", "meeting2.mp3"],
        help="Meeting audio file names to test together.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "voiceprint_demo_ecapa_multi",
        help="Combined output root directory.",
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
    parser.add_argument("--window-seconds", default=2.5, type=float, help="Speaker matching window length.")
    parser.add_argument("--hop-seconds", default=1.25, type=float, help="Speaker matching hop length.")
    parser.add_argument("--min-score", default=-1.0, type=float, help="Low-confidence score threshold.")
    parser.add_argument("--min-margin", default=0.005, type=float, help="Low-confidence margin threshold.")
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "speechbrain_ecapa", "mfcc"],
        default="speechbrain_ecapa",
        help="Speaker embedding backend.",
    )
    parser.add_argument(
        "--speaker-model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "speechbrain-spkrec-ecapa-voxceleb",
        help="Local SpeechBrain ECAPA model directory.",
    )
    parser.add_argument(
        "--strict-embedding-backend",
        action="store_true",
        help="Fail instead of falling back to MFCC when SpeechBrain ECAPA cannot be loaded.",
    )
    parser.add_argument(
        "--reuse-transcript",
        action="store_true",
        help="Reuse each meeting subdirectory's meeting.transcript.json if it already exists.",
    )
    parser.add_argument(
        "--reference-meeting",
        default="meeting.mp3",
        help="Meeting file that has a manual correction/reference transcript.",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=PROJECT_ROOT / "config" / "speaker_corrections.demo.json",
        help="Correction file used only for --reference-meeting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_inputs(data_dir, args.meetings)

    meeting_results: list[dict[str, Any]] = []
    for index, meeting in enumerate(args.meetings, start=1):
        meeting_path = data_dir / meeting
        meeting_id = Path(meeting).stem
        meeting_output_dir = output_dir / meeting_id
        meeting_output_dir.mkdir(parents=True, exist_ok=True)

        correction_path, has_manual_correction = resolve_correction_path(args, meeting, meeting_id, output_dir)

        print(f"[{index}/{len(args.meetings)}] Running demo for {meeting} ...", flush=True)
        run_single_meeting(args, meeting, meeting_output_dir, correction_path)
        result = read_json(meeting_output_dir / "final_demo_result.json")
        result["meeting_id"] = meeting_id
        result["meeting_file"] = meeting
        result["meeting_output_dir"] = str(meeting_output_dir)
        result["speaker_window_count"] = count_csv_rows(meeting_output_dir / "speaker_windows.csv")
        result["has_manual_correction"] = has_manual_correction
        meeting_results.append(result)

    combined = build_combined_result(args, data_dir, output_dir, meeting_results)
    write_json(combined, output_dir / "final_demo_result.json")
    write_json(combined, output_dir / "final_multi_meeting_result.json")
    write_summary_markdown(combined, output_dir / "multi_meeting_summary.md")

    print(f"Combined output: {output_dir}", flush=True)
    print(f"Combined final JSON: {output_dir / 'final_demo_result.json'}", flush=True)
    print(f"Summary: {output_dir / 'multi_meeting_summary.md'}", flush=True)
    return 0


def resolve_correction_path(
    args: argparse.Namespace,
    meeting: str,
    meeting_id: str,
    output_dir: Path,
) -> tuple[Path, bool]:
    per_meeting_path = PROJECT_ROOT / "config" / f"speaker_corrections.{meeting_id}.json"
    if per_meeting_path.exists():
        return per_meeting_path.resolve(), True

    default_correction = args.corrections.expanduser().resolve()
    if meeting == args.reference_meeting and default_correction.exists():
        return default_correction, True

    return output_dir / f"no_manual_correction_for_{meeting_id}.json", False


def validate_inputs(data_dir: Path, meetings: list[str]) -> None:
    required = [data_dir / meeting for meeting in meetings]
    required.extend(data_dir / f"{speaker}_solo.mp3" for speaker in ["A", "B", "C", "D"])
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required audio files:\n{joined}")


def run_single_meeting(
    args: argparse.Namespace,
    meeting: str,
    output_dir: Path,
    correction_path: Path,
) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_voiceprint_demo.py"),
        "--data-dir",
        str(args.data_dir.expanduser().resolve()),
        "--meeting",
        meeting,
        "--output-dir",
        str(output_dir),
        "--model",
        str(args.model),
        "--device",
        args.device,
        "--compute-type",
        args.compute_type,
        "--language",
        args.language,
        "--batch-size",
        str(args.batch_size),
        "--window-seconds",
        str(args.window_seconds),
        "--hop-seconds",
        str(args.hop_seconds),
        "--min-score",
        str(args.min_score),
        "--min-margin",
        str(args.min_margin),
        "--corrections",
        str(correction_path),
        "--embedding-backend",
        args.embedding_backend,
        "--speaker-model-path",
        str(args.speaker_model_path),
    ]
    if args.reuse_transcript:
        command.append("--reuse-transcript")
    if args.strict_embedding_backend:
        command.append("--strict-embedding-backend")

    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def build_combined_result(
    args: argparse.Namespace,
    data_dir: Path,
    output_dir: Path,
    meeting_results: list[dict[str, Any]],
) -> dict[str, Any]:
    total_duration = sum(float(item.get("meeting_duration_seconds") or 0.0) for item in meeting_results)
    total_windows = sum(int(item.get("speaker_window_count") or 0) for item in meeting_results)
    embedding_backend = meeting_results[0].get("embedding_backend", {}) if meeting_results else {}

    workflow_meetings = []
    all_segments = []
    for result in meeting_results:
        meeting_id = result["meeting_id"]
        segments = result.get("workflow_payload", {}).get("segments", [])
        enriched_segments = [
            {
                "meeting_id": meeting_id,
                "meeting_file": result["meeting_file"],
                **segment,
            }
            for segment in segments
        ]
        all_segments.extend(enriched_segments)
        workflow_meetings.append(
            {
                "meeting_id": meeting_id,
                "meeting_file": result["meeting_file"],
                "meeting_audio": result.get("meeting_audio"),
                "duration_seconds": result.get("meeting_duration_seconds"),
                "output_dir": result.get("meeting_output_dir"),
                "has_manual_correction": result.get("has_manual_correction", False),
                "segments": enriched_segments,
            }
        )

    return {
        "demo_name": "multi_meeting_voiceprint_demo",
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "meetings": [
            {
                "meeting_id": item["meeting_id"],
                "meeting_file": item["meeting_file"],
                "meeting_audio": item.get("meeting_audio"),
                "meeting_output_dir": item.get("meeting_output_dir"),
                "duration_seconds": item.get("meeting_duration_seconds"),
                "language": item.get("language"),
                "transcript_segment_count": len(item.get("transcript_segments", [])),
                "speaker_window_count": item.get("speaker_window_count"),
                "speaker_turn_count": len(item.get("speaker_turns", [])),
                "has_manual_correction": item.get("has_manual_correction", False),
                "correction_source": item.get("correction_source"),
            }
            for item in meeting_results
        ],
        "meeting_count": len(meeting_results),
        "total_duration_seconds": round(total_duration, 3),
        "total_speaker_window_count": total_windows,
        "embedding_backend": embedding_backend,
        "speaker_profiles": meeting_results[0].get("speaker_profiles", {}) if meeting_results else {},
        "workflow_payload": {
            "batch_id": "demo_batch_001",
            "participants": ["A", "B", "C", "D"],
            "meeting_count": len(meeting_results),
            "meetings": workflow_meetings,
            "all_segments": all_segments,
        },
        "note": (
            "已为存在人工校正文件的会议应用正文标注；"
            "没有人工正文标注的会议会保留 WhisperX + ECAPA 的自动识别结果。"
        ),
    }


def write_summary_markdown(payload: dict[str, Any], output_path: Path) -> None:
    backend = payload.get("embedding_backend", {})
    lines = [
        "# 双会议声纹识别 Demo 汇总",
        "",
        "## 基本信息",
        "",
        f"- 数据目录：`{payload['data_dir']}`",
        f"- 输出目录：`{payload['output_dir']}`",
        f"- 会议数量：`{payload['meeting_count']}`",
        f"- 总音频时长：`{payload['total_duration_seconds']}` 秒",
        f"- 总声纹窗口数：`{payload['total_speaker_window_count']}`",
        f"- 声纹后端：`{backend.get('name', 'unknown')}`",
    ]
    if backend.get("model_path"):
        lines.append(f"- 声纹模型路径：`{backend['model_path']}`")
    if backend.get("model_size_mb") is not None:
        lines.append(f"- 声纹模型目录大小：`{backend['model_size_mb']}` MB")

    lines.extend(
        [
            "",
            "## 会议测试结果",
            "",
            "| 会议 | 时长（秒） | 语言 | 转写片段数 | 声纹窗口数 | 说话人轮次 | 是否人工校正 | 输出目录 |",
            "|---|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for meeting in payload["meetings"]:
        lines.append(
            "| {meeting_file} | {duration:.3f} | {language} | {segments} | {windows} | {turns} | {corrected} | `{output}` |".format(
                meeting_file=meeting["meeting_file"],
                duration=float(meeting.get("duration_seconds") or 0.0),
                language=meeting.get("language") or "",
                segments=meeting.get("transcript_segment_count") or 0,
                windows=meeting.get("speaker_window_count") or 0,
                turns=meeting.get("speaker_turn_count") or 0,
                corrected="是" if meeting.get("has_manual_correction") else "否",
                output=meeting.get("meeting_output_dir") or "",
            )
        )

    lines.extend(
        [
            "",
            "## 输出说明",
            "",
            "- `final_demo_result.json`：双会议合并后的最终结构化输出。",
            "- `final_multi_meeting_result.json`：与 `final_demo_result.json` 内容相同，文件名更明确。",
            "- `workflow_payload.meetings`：按会议分组的结构化片段。",
            "- `workflow_payload.all_segments`：把两个会议的片段拍平成一个列表，每条片段都带 `meeting_id` 和 `meeting_file`。",
            "",
            "## 注意",
            "",
            str(payload.get("note", "")),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
