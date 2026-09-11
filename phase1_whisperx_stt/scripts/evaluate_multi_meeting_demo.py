from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from evaluate_demo import (  # noqa: E402
    SPEAKERS,
    build_confusion_matrix,
    build_duration_rows,
    build_runtime_rows,
    build_speaker_recall,
    char_error_rate,
    configure_font,
    confusion_accuracy,
    macro_recall,
    plot_confidence_curve,
    plot_confusion_matrix,
    plot_duration_bars,
    plot_similarity_curves,
    read_windows,
    token_error_rate,
    write_metrics_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate two meeting demos together using their reference transcripts."
    )
    parser.add_argument(
        "--multi-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "voiceprint_demo_ecapa_multi",
        help="Directory containing meeting/ and meeting2/ outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "voiceprint_demo_ecapa_multi" / "figures",
        help="Combined evaluation figure/report directory.",
    )
    parser.add_argument(
        "--meeting",
        action="append",
        nargs=3,
        metavar=("MEETING_ID", "DEMO_DIR", "REFERENCE_JSON"),
        help=(
            "One meeting spec. Can be repeated. Defaults to "
            "meeting + meeting2 under --multi-dir."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_font()

    multi_dir = args.multi_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(args, multi_dir)
    meeting_metrics = [evaluate_one_meeting(spec) for spec in specs]
    combined = build_combined_metrics(meeting_metrics, multi_dir)

    plot_combined_stt(combined, output_dir / "01_stt_cer_wer.png")
    plot_similarity_curves(combined["combined_windows"], output_dir / "02_voiceprint_similarity_curves.png")
    plot_confidence_curve(combined["combined_windows"], output_dir / "03_voiceprint_confidence_curve.png")
    plot_confusion_matrix(combined["confusion_matrix_raw"], output_dir / "04_speaker_confusion_matrix.png")
    plot_duration_bars(combined["duration_seconds"], output_dir / "05_speaker_duration_bar.png")
    plot_runtime_multi(combined, output_dir / "06_runtime_rtf_bar.png")

    metrics_for_json = {
        key: value
        for key, value in combined.items()
        if key not in {"combined_windows", "confusion_matrix_raw"}
    }
    metrics_for_json["confusion_matrix"] = {
        truth: dict(preds) for truth, preds in sorted(combined["confusion_matrix_raw"].items())
    }
    metrics_for_json["figures"] = [
        "01_stt_cer_wer.png",
        "02_voiceprint_similarity_curves.png",
        "03_voiceprint_confidence_curve.png",
        "04_speaker_confusion_matrix.png",
        "05_speaker_duration_bar.png",
        "06_runtime_rtf_bar.png",
    ]
    write_metrics_json(metrics_for_json, output_dir / "evaluation_metrics.json")
    write_report(metrics_for_json, output_dir / "evaluation_report.md")

    print(f"Combined figures: {output_dir}")
    print(f"Combined original CER: {combined['overall_stt_metrics'][0]['cer']:.4f}")
    print(f"Combined corrected CER: {combined['overall_stt_metrics'][1]['cer']:.4f}")
    print(f"Combined speaker accuracy excluding GROUP: {combined['speaker_accuracy_excluding_group']:.4f}")
    return 0


def build_specs(args: argparse.Namespace, multi_dir: Path) -> list[dict[str, Any]]:
    if args.meeting:
        return [
            {
                "meeting_id": item[0],
                "demo_dir": Path(item[1]).expanduser().resolve(),
                "reference": Path(item[2]).expanduser().resolve(),
            }
            for item in args.meeting
        ]

    return [
        {
            "meeting_id": "meeting",
            "demo_dir": multi_dir / "meeting",
            "reference": PROJECT_ROOT / "config" / "speaker_corrections.demo.json",
        },
        {
            "meeting_id": "meeting2",
            "demo_dir": multi_dir / "meeting2",
            "reference": PROJECT_ROOT / "config" / "speaker_corrections.meeting2.json",
        },
    ]


def evaluate_one_meeting(spec: dict[str, Any]) -> dict[str, Any]:
    meeting_id = str(spec["meeting_id"])
    demo_dir = Path(spec["demo_dir"])
    reference_path = Path(spec["reference"])

    reference = read_json(reference_path)
    original = read_json(demo_dir / "meeting.transcript.json")
    corrected = read_json(demo_dir / "meeting.corrected_transcript.json")
    final_result = read_json(demo_dir / "final_demo_result.json")
    windows = read_windows(demo_dir / "speaker_windows.csv")

    reference_text = "".join(segment["text"] for segment in reference["segments"])
    original_text = original.get("text", "")
    corrected_text = corrected.get("text", "")

    confusion = build_confusion_matrix(windows, reference["segments"])
    speaker_accuracy = confusion_accuracy(confusion, exclude_labels={"GROUP", "UNKNOWN"})
    speaker_recall = build_speaker_recall(confusion, SPEAKERS)
    runtime = build_runtime_rows(original, final_result)[0]

    return {
        "meeting_id": meeting_id,
        "demo_dir": str(demo_dir),
        "reference_path": str(reference_path),
        "meeting_audio": final_result.get("meeting_audio"),
        "duration_seconds": final_result.get("meeting_duration_seconds"),
        "language": final_result.get("language"),
        "embedding_backend": final_result.get("embedding_backend", {}),
        "reference_text": reference_text,
        "original_text": original_text,
        "corrected_text": corrected_text,
        "transcript_segment_count": len(final_result.get("transcript_segments", [])),
        "speaker_window_count": len(windows),
        "speaker_turn_count": len(final_result.get("speaker_turns", [])),
        "stt_metrics": [
            {
                "name": f"{meeting_id} WhisperX原始转写",
                "cer": char_error_rate(reference_text, original_text),
                "wer": token_error_rate(reference_text, original_text),
            },
            {
                "name": f"{meeting_id} 校正后转写",
                "cer": char_error_rate(reference_text, corrected_text),
                "wer": token_error_rate(reference_text, corrected_text),
            },
        ],
        "confusion_matrix_raw": confusion,
        "speaker_accuracy_excluding_group": speaker_accuracy,
        "speaker_recall": speaker_recall,
        "speaker_macro_recall": macro_recall(speaker_recall),
        "duration_seconds_by_speaker": build_duration_rows(final_result.get("speaker_turns", [])),
        "runtime": runtime,
        "windows": windows,
    }


def build_combined_metrics(meeting_metrics: list[dict[str, Any]], multi_dir: Path) -> dict[str, Any]:
    reference_text = "".join(item["reference_text"] for item in meeting_metrics)
    original_text = "".join(item["original_text"] for item in meeting_metrics)
    corrected_text = "".join(item["corrected_text"] for item in meeting_metrics)

    aggregate_confusion: dict[str, Counter[str]] = {}
    for item in meeting_metrics:
        for truth, preds in item["confusion_matrix_raw"].items():
            aggregate_confusion.setdefault(truth, Counter()).update(preds)

    speaker_accuracy = confusion_accuracy(aggregate_confusion, exclude_labels={"GROUP", "UNKNOWN"})
    speaker_recall = build_speaker_recall(aggregate_confusion, SPEAKERS)

    duration_counter: Counter[str] = Counter()
    for item in meeting_metrics:
        for row in item["duration_seconds_by_speaker"]:
            duration_counter[row["speaker"]] += float(row["seconds"])
    duration_rows = [
        {"speaker": speaker, "seconds": round(float(duration_counter.get(speaker, 0.0)), 3)}
        for speaker in ["A", "B", "C", "D", "GROUP"]
        if duration_counter.get(speaker, 0.0) > 0
    ]

    total_audio = sum(float(item["runtime"]["audio_seconds"]) for item in meeting_metrics)
    total_elapsed = sum(float(item["runtime"]["elapsed_seconds"]) for item in meeting_metrics)
    total_rtf = total_elapsed / total_audio if total_audio else 0.0

    combined_windows: list[dict[str, Any]] = []
    time_offset = 0.0
    for item in meeting_metrics:
        for row in item["windows"]:
            shifted = dict(row)
            shifted["start"] = float(row["start"]) + time_offset
            shifted["end"] = float(row["end"]) + time_offset
            shifted["meeting_id"] = item["meeting_id"]
            combined_windows.append(shifted)
        time_offset += float(item["duration_seconds"] or 0.0) + 5.0

    return {
        "evaluation_name": "multi_meeting_evaluation",
        "multi_dir": str(multi_dir),
        "meeting_count": len(meeting_metrics),
        "meetings": [
            {
                key: item[key]
                for key in [
                    "meeting_id",
                    "demo_dir",
                    "reference_path",
                    "meeting_audio",
                    "duration_seconds",
                    "language",
                    "transcript_segment_count",
                    "speaker_window_count",
                    "speaker_turn_count",
                    "stt_metrics",
                    "speaker_accuracy_excluding_group",
                    "speaker_recall",
                    "speaker_macro_recall",
                    "duration_seconds_by_speaker",
                    "runtime",
                ]
            }
            for item in meeting_metrics
        ],
        "embedding_backend": meeting_metrics[0].get("embedding_backend", {}) if meeting_metrics else {},
        "overall_stt_metrics": [
            {
                "name": "双会议 WhisperX原始转写",
                "cer": char_error_rate(reference_text, original_text),
                "wer": token_error_rate(reference_text, original_text),
            },
            {
                "name": "双会议 校正后转写",
                "cer": char_error_rate(reference_text, corrected_text),
                "wer": token_error_rate(reference_text, corrected_text),
            },
        ],
        "speaker_accuracy_excluding_group": speaker_accuracy,
        "speaker_recall": speaker_recall,
        "speaker_macro_recall": macro_recall(speaker_recall),
        "speaker_micro_recall_excluding_group": speaker_accuracy,
        "confusion_matrix_raw": aggregate_confusion,
        "duration_seconds": duration_rows,
        "runtime": {
            "item": "双会议 WhisperX转写",
            "elapsed_seconds": round(total_elapsed, 3),
            "audio_seconds": round(total_audio, 3),
            "rtf": round(total_rtf, 4),
        },
        "combined_windows": combined_windows,
    }


def plot_combined_stt(combined: dict[str, Any], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for item in combined["meetings"]:
        rows.extend(item["stt_metrics"])
    rows.extend(combined["overall_stt_metrics"])

    labels = [row["name"].replace(" WhisperX", "\nWhisperX").replace(" 校正", "\n校正") for row in rows]
    cer = [row["cer"] * 100 for row in rows]
    wer = [row["wer"] * 100 for row in rows]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.bar(x - width / 2, cer, width, label="CER")
    ax.bar(x + width / 2, wer, width, label="WER")
    ax.set_title("双会议 STT 转写错误率对比")
    ax.set_ylabel("错误率（%）")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    for values, offset in [(cer, -width / 2), (wer, width / 2)]:
        for i, value in enumerate(values):
            ax.text(i + offset, value + 1, f"{value:.1f}%", ha="center", fontsize=8)
    ax.set_ylim(0, max(cer + wer + [5]) * 1.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_runtime_multi(combined: dict[str, Any], output_path: Path) -> None:
    labels = [item["meeting_id"] for item in combined["meetings"]] + ["双会议合计"]
    audio_seconds = [float(item["runtime"]["audio_seconds"]) for item in combined["meetings"]]
    elapsed_seconds = [float(item["runtime"]["elapsed_seconds"]) for item in combined["meetings"]]
    rtf = [float(item["runtime"]["rtf"]) for item in combined["meetings"]]
    audio_seconds.append(float(combined["runtime"]["audio_seconds"]))
    elapsed_seconds.append(float(combined["runtime"]["elapsed_seconds"]))
    rtf.append(float(combined["runtime"]["rtf"]))

    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x - width / 2, audio_seconds, width, label="音频时长", color="#64748b")
    axes[0].bar(x + width / 2, elapsed_seconds, width, label="转写耗时", color="#2563eb")
    axes[0].set_title("音频时长与 WhisperX 转写耗时")
    axes[0].set_ylabel("秒")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()

    axes[1].bar(labels, rtf, color="#16a34a")
    axes[1].axhline(1.0, color="#ef4444", linestyle="--", linewidth=1, label="实时线 RTF=1")
    axes[1].set_title("Real Time Factor")
    axes[1].set_ylim(0, max(1.2, max(rtf + [0]) * 1.4))
    axes[1].legend()
    for index, value in enumerate(rtf):
        axes[1].text(index, value + 0.02, f"{value:.3f}", ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(metrics: dict[str, Any], output_path: Path) -> None:
    backend = metrics.get("embedding_backend", {})
    lines = [
        "# 双会议 Demo 评估图表报告",
        "",
        "## 图表清单",
        "",
        "| 文件 | 说明 |",
        "|---|---|",
        "| `01_stt_cer_wer.png` | 两个会议分别及合并后的 CER/WER 对比 |",
        "| `02_voiceprint_similarity_curves.png` | 双会议 A/B/C/D 声纹相似度时间曲线 |",
        "| `03_voiceprint_confidence_curve.png` | 双会议声纹识别置信度曲线 |",
        "| `04_speaker_confusion_matrix.png` | 双会议合并窗口级声纹混淆矩阵 |",
        "| `05_speaker_duration_bar.png` | 两个会议合并后的校正发言时长 |",
        "| `06_runtime_rtf_bar.png` | 两个会议分别及合并后的推理耗时和 RTF |",
        "",
        "## 总体关键指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for row in metrics["overall_stt_metrics"]:
        lines.append(f"| {row['name']} CER | {row['cer']:.4f} |")
        lines.append(f"| {row['name']} WER | {row['wer']:.4f} |")
    lines.append(f"| 声纹后端 | {backend.get('name', 'unknown')} |")
    if backend.get("model_size_mb") is not None:
        lines.append(f"| 声纹模型目录大小（MB） | {float(backend['model_size_mb']):.2f} |")
    lines.append(f"| 自动声纹窗口准确率（排除 GROUP） | {metrics['speaker_accuracy_excluding_group']:.4f} |")
    lines.append(f"| 自动声纹宏平均召回率 | {metrics['speaker_macro_recall']:.4f} |")
    lines.append(f"| 双会议 WhisperX RTF | {metrics['runtime']['rtf']:.4f} |")

    lines.extend(
        [
            "",
            "## 单会议指标",
            "",
            "| 会议 | 原始 CER | 原始 WER | 校正 CER | 校正 WER | 声纹准确率 | 宏平均召回率 | RTF |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in metrics["meetings"]:
        original, corrected = item["stt_metrics"]
        lines.append(
            f"| {item['meeting_id']} | {original['cer']:.4f} | {original['wer']:.4f} | "
            f"{corrected['cer']:.4f} | {corrected['wer']:.4f} | "
            f"{item['speaker_accuracy_excluding_group']:.4f} | {item['speaker_macro_recall']:.4f} | "
            f"{item['runtime']['rtf']:.4f} |"
        )

    lines.extend(["", "## 双会议自动声纹召回率", "", "| 说话人 | 正确窗口 | 总窗口 | 召回率 |", "|---|---:|---:|---:|"])
    for speaker, row in metrics["speaker_recall"].items():
        lines.append(f"| {speaker} | {row['correct_windows']} | {row['total_windows']} | {row['recall']:.4f} |")

    lines.extend(["", "## 双会议校正后发言时长", "", "| 说话人 | 秒 |", "|---|---:|"])
    for row in metrics["duration_seconds"]:
        lines.append(f"| {row['speaker']} | {row['seconds']:.2f} |")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "本报告使用两次会议的人工正文和说话人标注作为参考答案，因此可以计算 CER、WER、窗口级声纹准确率、召回率和混淆矩阵。",
            "校正后转写来自人工正文标注，因此 CER/WER 为 0，主要用于展示最终会议记录；原始 WhisperX 转写指标用于反映当前 tiny 模型在该音频上的实际转写质量。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
