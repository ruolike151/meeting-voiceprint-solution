from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from whisperx_stt.audio import get_audio_duration_seconds


SPEAKERS = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and plot the four-person demo.")
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "voiceprint_demo",
        help="Directory containing demo outputs.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / "config" / "speaker_corrections.demo.json",
        help="Reference transcript with speaker labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "voiceprint_demo" / "figures",
        help="Figure and evaluation report output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_font()

    demo_dir = args.demo_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = json.loads(args.reference.expanduser().resolve().read_text(encoding="utf-8"))
    original = json.loads((demo_dir / "meeting.transcript.json").read_text(encoding="utf-8"))
    corrected = json.loads((demo_dir / "meeting.corrected_transcript.json").read_text(encoding="utf-8"))
    final_result = json.loads((demo_dir / "final_demo_result.json").read_text(encoding="utf-8"))
    windows = read_windows(demo_dir / "speaker_windows.csv")
    embedding_backend = final_result.get("embedding_backend", {})

    reference_text = "".join(segment["text"] for segment in reference["segments"])
    original_text = original.get("text", "")
    corrected_text = corrected.get("text", "")

    stt_metrics = [
        {
            "name": "WhisperX原始转写",
            "cer": char_error_rate(reference_text, original_text),
            "wer": token_error_rate(reference_text, original_text),
        },
        {
            "name": "校正后转写",
            "cer": char_error_rate(reference_text, corrected_text),
            "wer": token_error_rate(reference_text, corrected_text),
        },
    ]

    reference_segments = reference["segments"]
    confusion = build_confusion_matrix(windows, reference_segments)
    speaker_accuracy = confusion_accuracy(confusion, exclude_labels={"GROUP", "UNKNOWN"})
    speaker_recall = build_speaker_recall(confusion, SPEAKERS)
    duration_rows = build_duration_rows(final_result["speaker_turns"])
    runtime_rows = build_runtime_rows(original, final_result)

    plot_stt_errors(stt_metrics, output_dir / "01_stt_cer_wer.png")
    plot_similarity_curves(windows, output_dir / "02_voiceprint_similarity_curves.png")
    plot_confidence_curve(windows, output_dir / "03_voiceprint_confidence_curve.png")
    plot_confusion_matrix(confusion, output_dir / "04_speaker_confusion_matrix.png")
    plot_duration_bars(duration_rows, output_dir / "05_speaker_duration_bar.png")
    plot_runtime(runtime_rows, output_dir / "06_runtime_rtf_bar.png")

    write_metrics_json(
        {
            "reference_path": str(args.reference.expanduser().resolve()),
            "demo_dir": str(demo_dir),
            "embedding_backend": embedding_backend,
            "stt_metrics": stt_metrics,
            "speaker_accuracy_excluding_group": speaker_accuracy,
            "speaker_recall": speaker_recall,
            "speaker_macro_recall": macro_recall(speaker_recall),
            "speaker_micro_recall_excluding_group": speaker_accuracy,
            "confusion_matrix": {
                truth: dict(preds) for truth, preds in sorted(confusion.items())
            },
            "duration_seconds": duration_rows,
            "runtime": runtime_rows,
            "figures": [
                "01_stt_cer_wer.png",
                "02_voiceprint_similarity_curves.png",
                "03_voiceprint_confidence_curve.png",
                "04_speaker_confusion_matrix.png",
                "05_speaker_duration_bar.png",
                "06_runtime_rtf_bar.png",
            ],
        },
        output_dir / "evaluation_metrics.json",
    )
    write_report(
        stt_metrics,
        speaker_accuracy,
        speaker_recall,
        duration_rows,
        runtime_rows,
        embedding_backend,
        output_dir / "evaluation_report.md",
    )

    print(f"Figures: {output_dir}")
    print(f"Original CER: {stt_metrics[0]['cer']:.4f}")
    print(f"Corrected CER: {stt_metrics[1]['cer']:.4f}")
    print(f"Speaker accuracy excluding GROUP: {speaker_accuracy:.4f}")
    return 0


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


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？、；：,.!?;:\-—“”\"'（）()《》<>【】\[\]`]", "", text)
    return text.lower()


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            cost = 0 if left == right else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(normalize_text(reference))
    hyp = list(normalize_text(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def token_error_rate(reference: str, hypothesis: str) -> float:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


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


def speaker_at(time_point: float, segments: list[dict[str, Any]]) -> str:
    for segment in segments:
        if float(segment["start"]) <= time_point < float(segment["end"]):
            return str(segment["speaker"])
    return "UNKNOWN"


def build_confusion_matrix(
    windows: list[dict[str, Any]],
    reference_segments: list[dict[str, Any]],
) -> dict[str, Counter[str]]:
    confusion: dict[str, Counter[str]] = {}
    for window in windows:
        midpoint = (window["start"] + window["end"]) / 2.0
        truth = speaker_at(midpoint, reference_segments)
        pred = str(window["speaker"])
        confusion.setdefault(truth, Counter())[pred] += 1
    return confusion


def confusion_accuracy(confusion: dict[str, Counter[str]], exclude_labels: set[str]) -> float:
    correct = 0
    total = 0
    for truth, preds in confusion.items():
        if truth in exclude_labels:
            continue
        for pred, count in preds.items():
            total += count
            if pred == truth:
                correct += count
    return correct / total if total else 0.0


def build_speaker_recall(
    confusion: dict[str, Counter[str]],
    speakers: list[str],
) -> dict[str, dict[str, Any]]:
    recall_rows: dict[str, dict[str, Any]] = {}
    for speaker in speakers:
        preds = confusion.get(speaker, Counter())
        support = sum(preds.values())
        correct = preds.get(speaker, 0)
        recall_rows[speaker] = {
            "correct_windows": int(correct),
            "total_windows": int(support),
            "recall": correct / support if support else 0.0,
        }
    return recall_rows


def macro_recall(speaker_recall: dict[str, dict[str, Any]]) -> float:
    recalls = [float(row["recall"]) for row in speaker_recall.values() if row["total_windows"] > 0]
    return float(np.mean(recalls)) if recalls else 0.0


def build_duration_rows(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seconds: Counter[str] = Counter()
    for turn in turns:
        seconds[str(turn["speaker"])] += float(turn["end"]) - float(turn["start"])
    return [
        {"speaker": speaker, "seconds": round(float(seconds.get(speaker, 0.0)), 3)}
        for speaker in ["A", "B", "C", "D", "GROUP"]
        if seconds.get(speaker, 0.0) > 0
    ]


def build_runtime_rows(original: dict[str, Any], final_result: dict[str, Any]) -> list[dict[str, Any]]:
    duration = float(final_result.get("meeting_duration_seconds") or 0.0)
    elapsed = float(original.get("elapsed_seconds") or 0.0)
    rtf = elapsed / duration if duration else 0.0
    return [
        {
            "item": "WhisperX转写",
            "elapsed_seconds": round(elapsed, 3),
            "audio_seconds": round(duration, 3),
            "rtf": round(rtf, 4),
        }
    ]


def plot_stt_errors(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [row["name"] for row in rows]
    cer = [row["cer"] * 100 for row in rows]
    wer = [row["wer"] * 100 for row in rows]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, cer, width, label="CER")
    ax.bar(x + width / 2, wer, width, label="WER")
    ax.set_title("STT 转写错误率对比")
    ax.set_ylabel("错误率（%）")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for values, offset in [(cer, -width / 2), (wer, width / 2)]:
        for i, value in enumerate(values):
            ax.text(i + offset, value + 1, f"{value:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, max(cer + wer + [5]) * 1.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_similarity_curves(windows: list[dict[str, Any]], output_path: Path) -> None:
    times = [(row["start"] + row["end"]) / 2.0 for row in windows]
    fig, ax = plt.subplots(figsize=(11, 5))
    values: list[float] = []
    for speaker in SPEAKERS:
        speaker_values = [row[speaker] for row in windows]
        values.extend(speaker_values)
        ax.plot(times, speaker_values, label=speaker, linewidth=1.8)
    ax.set_title("A/B/C/D 声纹相似度时间曲线")
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("余弦相似度")
    if values:
        y_min = min(values)
        y_max = max(values)
        padding = max(0.02, (y_max - y_min) * 0.18)
        ax.set_ylim(max(-1.0, y_min - padding), min(1.0, y_max + padding))
    ax.grid(alpha=0.25)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confidence_curve(windows: list[dict[str, Any]], output_path: Path) -> None:
    times = [(row["start"] + row["end"]) / 2.0 for row in windows]
    confidence = [row["confidence"] for row in windows]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(times, confidence, color="#2563eb", linewidth=2)
    ax.scatter(times, confidence, c=[speaker_color(row["speaker"]) for row in windows], s=22)
    ax.set_title("声纹识别置信度随时间变化")
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


def speaker_color(speaker: str) -> str:
    return {
        "A": "#2563eb",
        "B": "#16a34a",
        "C": "#f59e0b",
        "D": "#dc2626",
    }.get(speaker, "#6b7280")


def plot_confusion_matrix(confusion: dict[str, Counter[str]], output_path: Path) -> None:
    labels = ["A", "B", "C", "D", "GROUP", "UNKNOWN"]
    labels = [label for label in labels if label in confusion or any(label in c for c in confusion.values())]
    matrix = np.array([[confusion.get(truth, Counter()).get(pred, 0) for pred in SPEAKERS] for truth in labels])
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("自动声纹匹配混淆矩阵（窗口级）")
    ax.set_xlabel("预测说话人")
    ax.set_ylabel("真实说话人")
    ax.set_xticks(np.arange(len(SPEAKERS)))
    ax.set_xticklabels(SPEAKERS)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="#111827")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_duration_bars(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [row["speaker"] for row in rows]
    seconds = [row["seconds"] for row in rows]
    colors = [speaker_color(label) for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, seconds, color=colors)
    ax.set_title("校正后各说话人发言时长")
    ax.set_xlabel("说话人")
    ax.set_ylabel("发言时长（秒）")
    for index, value in enumerate(seconds):
        ax.text(index, value + 0.5, f"{value:.1f}s", ha="center")
    ax.set_ylim(0, max(seconds + [5]) * 1.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_runtime(rows: list[dict[str, Any]], output_path: Path) -> None:
    row = rows[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    axes[0].bar(["音频时长", "推理耗时"], [row["audio_seconds"], row["elapsed_seconds"]], color=["#64748b", "#2563eb"])
    axes[0].set_title("推理耗时对比")
    axes[0].set_ylabel("秒")
    for index, value in enumerate([row["audio_seconds"], row["elapsed_seconds"]]):
        axes[0].text(index, value + 1, f"{value:.1f}s", ha="center")
    axes[1].bar(["RTF"], [row["rtf"]], color="#16a34a")
    axes[1].axhline(1.0, color="#ef4444", linestyle="--", linewidth=1, label="实时线 RTF=1")
    axes[1].set_title("Real Time Factor")
    axes[1].set_ylim(0, max(1.2, row["rtf"] * 1.4))
    axes[1].legend()
    axes[1].text(0, row["rtf"] + 0.03, f"{row['rtf']:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_metrics_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(
    stt_metrics: list[dict[str, Any]],
    speaker_accuracy: float,
    speaker_recall: dict[str, dict[str, Any]],
    duration_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    embedding_backend: dict[str, Any],
    output_path: Path,
) -> None:
    lines = [
        "# 四人会议 Demo 评估图表报告",
        "",
        "## 图表清单",
        "",
        "| 文件 | 说明 |",
        "|---|---|",
        "| `01_stt_cer_wer.png` | WhisperX 原始转写与校正后转写的 CER/WER 对比 |",
        "| `02_voiceprint_similarity_curves.png` | A/B/C/D 声纹相似度随时间变化 |",
        "| `03_voiceprint_confidence_curve.png` | 声纹识别置信度随时间变化 |",
        "| `04_speaker_confusion_matrix.png` | 自动声纹匹配与参考标注的窗口级混淆矩阵 |",
        "| `05_speaker_duration_bar.png` | 校正后各说话人发言时长 |",
        "| `06_runtime_rtf_bar.png` | 音频时长、推理耗时和 RTF |",
        "",
        "## 关键指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for row in stt_metrics:
        lines.append(f"| {row['name']} CER | {row['cer']:.4f} |")
        lines.append(f"| {row['name']} WER | {row['wer']:.4f} |")
    if embedding_backend:
        lines.append(f"| 声纹后端 | {embedding_backend.get('name', 'unknown')} |")
        if embedding_backend.get("model_size_mb") is not None:
            lines.append(f"| 声纹模型目录大小（MB） | {float(embedding_backend['model_size_mb']):.2f} |")
    lines.append(f"| 自动声纹窗口准确率（排除 GROUP） | {speaker_accuracy:.4f} |")
    lines.append(f"| 自动声纹宏平均召回率 | {macro_recall(speaker_recall):.4f} |")
    lines.append(f"| WhisperX RTF | {runtime_rows[0]['rtf']:.4f} |")

    lines.extend(["", "## 自动声纹召回率", "", "| 说话人 | 正确窗口 | 总窗口 | 召回率 |", "|---|---:|---:|---:|"])
    for speaker, row in speaker_recall.items():
        lines.append(
            f"| {speaker} | {row['correct_windows']} | {row['total_windows']} | {row['recall']:.4f} |"
        )

    lines.extend(["", "## 校正后发言时长", "", "| 说话人 | 秒 |", "|---|---:|"])
    for row in duration_rows:
        lines.append(f"| {row['speaker']} | {row['seconds']:.2f} |")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "当前项目没有训练过程，因此没有 loss 曲线。这里输出的是推理阶段更合理的评估图：错误率、相似度、置信度、混淆矩阵、发言时长和 RTF。",
            "校正后转写使用用户提供的会议正文作为参考标注，因此可作为 demo 展示结果；自动声纹混淆矩阵保留 ECAPA 后端的原始窗口级匹配效果，便于定位哪些员工或片段仍需要优化。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
