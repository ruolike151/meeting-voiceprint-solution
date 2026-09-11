from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEAKERS = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate no-reference evaluation figures for one or more voiceprint demo outputs."
    )
    parser.add_argument(
        "--demo-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "voiceprint_demo_ecapa_multi",
        help="Demo output root. Can be a multi-meeting root or a single-meeting output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "voiceprint_demo_ecapa_multi" / "unlabeled_evaluation",
        help="Output directory for no-reference evaluation report and figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_font()

    demo_root = args.demo_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meeting_dirs = discover_meeting_dirs(demo_root)
    if not meeting_dirs:
        raise FileNotFoundError(f"No meeting output directories found under: {demo_root}")

    meeting_reports = []
    for meeting_dir in meeting_dirs:
        meeting_report = evaluate_meeting(meeting_dir, output_dir)
        meeting_reports.append(meeting_report)

    payload = build_payload(demo_root, output_dir, meeting_reports)
    write_json(payload, output_dir / "unlabeled_evaluation_metrics.json")
    write_csv_summary(meeting_reports, output_dir / "unlabeled_evaluation_summary.csv")
    plot_rtf(meeting_reports, output_dir / "00_runtime_rtf_summary.png")
    write_report(payload, output_dir / "unlabeled_evaluation_report.md")

    print(f"Unlabeled evaluation output: {output_dir}")
    print(f"Report: {output_dir / 'unlabeled_evaluation_report.md'}")
    return 0


def discover_meeting_dirs(demo_root: Path) -> list[Path]:
    if (demo_root / "speaker_windows.csv").exists() and (demo_root / "final_demo_result.json").exists():
        return [demo_root]
    return sorted(
        path
        for path in demo_root.iterdir()
        if path.is_dir()
        and (path / "speaker_windows.csv").exists()
        and (path / "final_demo_result.json").exists()
    )


def evaluate_meeting(meeting_dir: Path, output_dir: Path) -> dict[str, Any]:
    final_result = read_json(meeting_dir / "final_demo_result.json")
    transcript = read_json(meeting_dir / "meeting.transcript.json")
    windows = read_windows(meeting_dir / "speaker_windows.csv")

    meeting_id = meeting_dir.name
    duration = float(final_result.get("meeting_duration_seconds") or 0.0)
    elapsed = float(transcript.get("elapsed_seconds") or 0.0)
    rtf = elapsed / duration if duration else 0.0
    predicted_turns = merge_windows_to_turns(windows, max_gap_seconds=1.25)
    predicted_durations = build_duration_rows(predicted_turns)

    plot_similarity_curves(
        windows,
        output_dir / f"{meeting_id}_01_voiceprint_similarity_curves.png",
        title=f"{meeting_id} 声纹相似度时间曲线",
    )
    plot_confidence_curve(
        windows,
        output_dir / f"{meeting_id}_02_voiceprint_confidence_curve.png",
        title=f"{meeting_id} 声纹识别置信度时间曲线",
    )
    plot_predicted_duration_bars(
        predicted_durations,
        output_dir / f"{meeting_id}_03_predicted_speaker_duration.png",
        title=f"{meeting_id} 自动预测发言时长",
    )

    return {
        "meeting_id": meeting_id,
        "meeting_audio": final_result.get("meeting_audio"),
        "language": final_result.get("language"),
        "duration_seconds": round(duration, 3),
        "transcription_elapsed_seconds": round(elapsed, 3),
        "rtf": round(rtf, 4),
        "speaker_window_count": len(windows),
        "predicted_turn_count": len(predicted_turns),
        "predicted_duration_seconds": predicted_durations,
        "embedding_backend": final_result.get("embedding_backend", {}),
        "figures": [
            f"{meeting_id}_01_voiceprint_similarity_curves.png",
            f"{meeting_id}_02_voiceprint_confidence_curve.png",
            f"{meeting_id}_03_predicted_speaker_duration.png",
        ],
        "not_computed_without_reference": [
            "STT 转写准确率",
            "CER / WER",
            "声纹识别准确率",
            "混淆矩阵",
        ],
    }


def build_payload(
    demo_root: Path,
    output_dir: Path,
    meeting_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    total_duration = sum(float(row["duration_seconds"]) for row in meeting_reports)
    total_elapsed = sum(float(row["transcription_elapsed_seconds"]) for row in meeting_reports)
    total_windows = sum(int(row["speaker_window_count"]) for row in meeting_reports)
    return {
        "evaluation_stage": "未使用会议原文核对前的无标注评估",
        "demo_root": str(demo_root),
        "output_dir": str(output_dir),
        "meeting_count": len(meeting_reports),
        "total_duration_seconds": round(total_duration, 3),
        "total_transcription_elapsed_seconds": round(total_elapsed, 3),
        "overall_rtf": round(total_elapsed / total_duration, 4) if total_duration else 0.0,
        "total_speaker_window_count": total_windows,
        "meetings": meeting_reports,
        "available_without_reference": [
            "推理速度 / RTF",
            "声纹相似度曲线",
            "声纹置信度曲线",
            "自动说话人时间轴",
            "自动预测发言时长",
        ],
        "requires_reference": {
            "STT 转写准确率": "需要人工整理的会议原文",
            "CER / WER 曲线": "需要人工整理的会议原文，最好是简体文本",
            "声纹识别是否认对人": "需要每段时间对应哪个真实说话人",
            "混淆矩阵": "需要真实说话人标签",
        },
    }


def read_windows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {
                "start": float(row["start"]),
                "end": float(row["end"]),
                "speaker": row["speaker"],
                "score": float(row["score"]),
                "margin": float(row["margin"]),
                "confidence": float(row["confidence"]),
                "status": row["status"],
            }
            for speaker in SPEAKERS:
                parsed[speaker] = float(row[speaker])
            rows.append(parsed)
    return rows


def merge_windows_to_turns(
    windows: list[dict[str, Any]],
    max_gap_seconds: float,
) -> list[dict[str, Any]]:
    if not windows:
        return []

    turns: list[dict[str, Any]] = []
    current = {
        "start": float(windows[0]["start"]),
        "end": float(windows[0]["end"]),
        "speaker": str(windows[0]["speaker"]),
        "scores": [float(windows[0]["score"])],
        "confidences": [float(windows[0]["confidence"])],
    }
    for window in windows[1:]:
        speaker = str(window["speaker"])
        start = float(window["start"])
        end = float(window["end"])
        same_speaker = speaker == current["speaker"]
        close_enough = start <= float(current["end"]) + max_gap_seconds
        if same_speaker and close_enough:
            current["end"] = max(float(current["end"]), end)
            current["scores"].append(float(window["score"]))
            current["confidences"].append(float(window["confidence"]))
        else:
            turns.append(finalize_turn(current))
            current = {
                "start": start,
                "end": end,
                "speaker": speaker,
                "scores": [float(window["score"])],
                "confidences": [float(window["confidence"])],
            }
    turns.append(finalize_turn(current))
    return turns


def finalize_turn(turn: dict[str, Any]) -> dict[str, Any]:
    scores = turn.pop("scores")
    confidences = turn.pop("confidences")
    return {
        "start": round(float(turn["start"]), 3),
        "end": round(float(turn["end"]), 3),
        "speaker": str(turn["speaker"]),
        "avg_score": round(float(np.mean(scores)), 4),
        "avg_confidence": round(float(np.mean(confidences)), 4),
    }


def build_duration_rows(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seconds: Counter[str] = Counter()
    for turn in turns:
        seconds[str(turn["speaker"])] += max(0.0, float(turn["end"]) - float(turn["start"]))
    return [
        {"speaker": speaker, "seconds": round(float(seconds.get(speaker, 0.0)), 3)}
        for speaker in SPEAKERS
        if seconds.get(speaker, 0.0) > 0
    ]


def configure_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    plt.rcParams["font.sans-serif"] = candidates + plt.rcParams.get("font.sans-serif", [])


def plot_similarity_curves(windows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    times = [(row["start"] + row["end"]) / 2.0 for row in windows]
    fig, ax = plt.subplots(figsize=(11, 5))
    values: list[float] = []
    for speaker in SPEAKERS:
        speaker_values = [row[speaker] for row in windows]
        values.extend(speaker_values)
        ax.plot(times, speaker_values, label=speaker, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("余弦相似度")
    if values:
        y_min = min(values)
        y_max = max(values)
        padding = max(0.03, (y_max - y_min) * 0.18)
        ax.set_ylim(max(-1.0, y_min - padding), min(1.0, y_max + padding))
    ax.grid(alpha=0.25)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confidence_curve(windows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    times = [(row["start"] + row["end"]) / 2.0 for row in windows]
    confidence = [row["confidence"] for row in windows]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(times, confidence, color="#2563eb", linewidth=1.8)
    ax.scatter(times, confidence, c=[speaker_color(row["speaker"]) for row in windows], s=16)
    ax.set_title(title)
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("置信度")
    if confidence:
        y_min = min(confidence)
        y_max = max(confidence)
        padding = max(0.03, (y_max - y_min) * 0.2)
        ax.set_ylim(max(0.0, y_min - padding), min(1.0, y_max + padding))
    ax.grid(alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=s, markerfacecolor=speaker_color(s), markersize=8)
        for s in SPEAKERS
    ]
    ax.legend(handles=handles, title="预测说话人", ncol=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_predicted_duration_bars(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    labels = [row["speaker"] for row in rows]
    seconds = [row["seconds"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, seconds, color=[speaker_color(label) for label in labels])
    ax.set_title(title)
    ax.set_xlabel("预测说话人")
    ax.set_ylabel("预测发言时长（秒）")
    for index, value in enumerate(seconds):
        ax.text(index, value + max(seconds + [1]) * 0.02, f"{value:.1f}s", ha="center")
    ax.set_ylim(0, max(seconds + [5]) * 1.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rtf(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [row["meeting_id"] for row in rows]
    rtfs = [float(row["rtf"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, rtfs, color="#16a34a")
    ax.axhline(1.0, color="#ef4444", linestyle="--", linewidth=1, label="实时线 RTF=1")
    ax.set_title("WhisperX 推理速度 RTF 汇总")
    ax.set_xlabel("会议")
    ax.set_ylabel("RTF")
    ax.set_ylim(0, max(1.2, max(rtfs + [0]) * 1.4))
    for index, value in enumerate(rtfs):
        ax.text(index, value + 0.03, f"{value:.4f}", ha="center")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def speaker_color(speaker: str) -> str:
    return {
        "A": "#2563eb",
        "B": "#16a34a",
        "C": "#f59e0b",
        "D": "#dc2626",
    }.get(speaker, "#6b7280")


def write_report(payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# 未使用会议原文核对前的自动评估报告",
        "",
        "## 评估边界",
        "",
        "本报告不使用会议原文和真实说话人时间标注，因此只评估工程运行状态和模型自动输出分布。",
        "CER、WER、声纹识别准确率和混淆矩阵需要人工标准答案，本报告不计算这些指标。",
        "",
        "## 可输出评估内容",
        "",
        "| 评估项 | 是否需要会议原文 | 本报告是否输出 |",
        "|---|---|---|",
        "| 推理速度 / RTF | 不需要 | 是 |",
        "| 声纹相似度曲线 | 不需要 | 是 |",
        "| 声纹置信度曲线 | 不需要 | 是 |",
        "| 自动预测发言时长 | 不需要，但只是预测值 | 是 |",
        "| STT 转写准确率 | 需要 | 否 |",
        "| CER / WER 曲线 | 需要 | 否 |",
        "| 声纹识别是否认对人 | 需要真实说话人时间标注 | 否 |",
        "| 混淆矩阵 | 需要真实说话人标签 | 否 |",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 会议数量 | {payload['meeting_count']} |",
        f"| 总音频时长（秒） | {payload['total_duration_seconds']:.3f} |",
        f"| 总 WhisperX 推理耗时（秒） | {payload['total_transcription_elapsed_seconds']:.3f} |",
        f"| 整体 RTF | {payload['overall_rtf']:.4f} |",
        f"| 总声纹窗口数 | {payload['total_speaker_window_count']} |",
        "",
        "## 分会议指标",
        "",
        "| 会议 | 音频时长（秒） | WhisperX耗时（秒） | RTF | 声纹窗口数 | 自动预测轮次 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for meeting in payload["meetings"]:
        lines.append(
            f"| {meeting['meeting_id']} | {meeting['duration_seconds']:.3f} | "
            f"{meeting['transcription_elapsed_seconds']:.3f} | {meeting['rtf']:.4f} | "
            f"{meeting['speaker_window_count']} | {meeting['predicted_turn_count']} |"
        )

    lines.extend(["", "## 自动预测发言时长", ""])
    for meeting in payload["meetings"]:
        lines.extend(
            [
                f"### {meeting['meeting_id']}",
                "",
                "| 预测说话人 | 预测发言时长（秒） |",
                "|---|---:|",
            ]
        )
        for row in meeting["predicted_duration_seconds"]:
            lines.append(f"| {row['speaker']} | {row['seconds']:.3f} |")
        lines.append("")

    lines.extend(
        [
            "## 图表清单",
            "",
            "| 文件 | 说明 |",
            "|---|---|",
            "| `00_runtime_rtf_summary.png` | 两段会议 WhisperX 推理 RTF 对比 |",
        ]
    )
    for meeting in payload["meetings"]:
        for figure in meeting["figures"]:
            lines.append(f"| `{figure}` | {meeting['meeting_id']} 无标注自动评估图 |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "meeting_id",
                "meeting_audio",
                "duration_seconds",
                "transcription_elapsed_seconds",
                "rtf",
                "speaker_window_count",
                "predicted_turn_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in writer.fieldnames})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
