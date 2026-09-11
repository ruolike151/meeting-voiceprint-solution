# Phase 1：WhisperX STT + 员工声纹识别小 Demo

本目录用于完成实习任务第一阶段和四人会议小 demo：

```text
会议录音 -> WhisperX 语音转文本 -> 单人注册录音建声纹库 -> 会议窗口声纹匹配 -> 说话人身份对齐 -> Agent Workflow 结构化输出
```

当前推荐路线已经落地：声纹后端优先使用 `SpeechBrain ECAPA-TDNN` 预训练 speaker embedding，MFCC 仅作为模型缺失或加载失败时的离线兜底。

## 1. 当前环境

- 项目目录：`F:\实习\voiceprint_solution_work\phase1_whisperx_stt`
- 虚拟环境：`F:\实习\voiceprint_solution_work\phase1_whisperx_stt\.venv`
- Python：3.12.9
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch：`2.8.0+cu128`
- Torchaudio：`2.8.0+cu128`
- WhisperX：`3.8.6`
- SpeechBrain：`1.1.0`
- 声纹模型：`speechbrain/spkrec-ecapa-voxceleb`
- 本地声纹模型目录：`F:\实习\voiceprint_solution_work\phase1_whisperx_stt\models\speechbrain-spkrec-ecapa-voxceleb`
- ECAPA 声纹模型目录大小：约 `84.86 MB`
- ECAPA embedding 维度：`192`

进入虚拟环境：

```powershell
cd F:\实习\voiceprint_solution_work\phase1_whisperx_stt
.\.venv\Scripts\Activate.ps1
```

检查环境：

```powershell
python .\scripts\env_check.py
```

## 2. 输入数据要求

当前 demo 使用的数据目录：

```text
F:\实习\voiceprint_solution_work\data
```

需要包含：

| 文件 | 用途 | 当前检测时长 |
|---|---|---:|
| `meeting.mp3` | 四人会议完整录音 | 86.52 秒 |
| `A_solo.mp3` | 员工 A 注册录音 | 22.416 秒 |
| `B_solo.mp3` | 员工 B 注册录音 | 17.760 秒 |
| `C_solo.mp3` | 员工 C 注册录音 | 21.720 秒 |
| `D_solo.mp3` | 员工 D 注册录音 | 18.000 秒 |

说明：

- `meeting.mp3` 应该是一整个会议录音，不是一个人一段。
- `A_solo.mp3` 到 `D_solo.mp3` 是每位员工各自的单独注册录音。
- 小 demo 中每位员工注册录音 15 秒以上已经可用；更稳妥的注册建议是 30 到 60 秒、环境尽量接近会议场景。

## 3. 运行 ECAPA 推荐版 Demo

首次完整运行：

```powershell
python .\scripts\run_voiceprint_demo.py `
  --data-dir ..\data `
  --output-dir .\outputs\voiceprint_demo_ecapa `
  --model .\models\faster-whisper-tiny `
  --device cuda `
  --compute-type float16 `
  --language auto `
  --batch-size 8 `
  --corrections .\config\speaker_corrections.demo.json `
  --embedding-backend speechbrain_ecapa `
  --speaker-model-path .\models\speechbrain-spkrec-ecapa-voxceleb `
  --strict-embedding-backend
```

如果已经有 `meeting.transcript.json`，只想重新跑声纹匹配和输出整理：

```powershell
python .\scripts\run_voiceprint_demo.py `
  --data-dir ..\data `
  --output-dir .\outputs\voiceprint_demo_ecapa `
  --model .\models\faster-whisper-tiny `
  --device cuda `
  --compute-type float16 `
  --language auto `
  --batch-size 8 `
  --reuse-transcript `
  --corrections .\config\speaker_corrections.demo.json `
  --embedding-backend speechbrain_ecapa `
  --speaker-model-path .\models\speechbrain-spkrec-ecapa-voxceleb `
  --strict-embedding-backend
```

如果希望模型缺失时也能跑通演示，可去掉 `--strict-embedding-backend`，脚本会自动回退到 MFCC：

```powershell
python .\scripts\run_voiceprint_demo.py --embedding-backend auto
```

## 4. 生成评估图表

```powershell
python .\scripts\evaluate_demo.py `
  --demo-dir .\outputs\voiceprint_demo_ecapa `
  --reference .\config\speaker_corrections.demo.json `
  --output-dir .\outputs\voiceprint_demo_ecapa\figures
```

## 5. 双会议批量测试

如果数据目录中同时有 `meeting.mp3` 和 `meeting2.mp3`，可以运行批量测试脚本：

```powershell
python .\scripts\run_multi_meeting_demo.py `
  --data-dir ..\data `
  --meetings meeting.mp3 meeting2.mp3 `
  --output-dir .\outputs\voiceprint_demo_ecapa_multi `
  --model .\models\faster-whisper-tiny `
  --device cuda `
  --compute-type float16 `
  --language auto `
  --batch-size 8 `
  --reuse-transcript `
  --embedding-backend speechbrain_ecapa `
  --speaker-model-path .\models\speechbrain-spkrec-ecapa-voxceleb `
  --strict-embedding-backend
```

批量测试输出目录：

```text
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\outputs\voiceprint_demo_ecapa_multi
```

核心输出：

| 文件/目录 | 说明 |
|---|---|
| `meeting/` | `meeting.mp3` 的单会议完整输出 |
| `meeting2/` | `meeting2.mp3` 的单会议完整输出 |
| `final_demo_result.json` | 两个会议合并后的最终结构化输出 |
| `final_multi_meeting_result.json` | 与 `final_demo_result.json` 内容相同，文件名更明确 |
| `multi_meeting_summary.md` | 双会议测试汇总说明 |

本次双会议测试结果：

| 会议 | 时长（秒） | 语言 | 转写片段数 | 声纹窗口数 | 说话人轮次 | 是否人工校正 |
|---|---:|---|---:|---:|---:|---|
| `meeting.mp3` | 86.520 | zh | 7 | 69 | 7 | 是 |
| `meeting2.mp3` | 357.984 | zh | 24 | 286 | 24 | 是 |

说明：`meeting.mp3` 使用 `speaker_corrections.demo.json`；`meeting2.mp3` 使用 `speaker_corrections.meeting2.json`。两个会议的最终展示文本和人物标签都已应用人工正文校正。

双会议批量结果生成后，可以分别输出两个会议各自的评估图表：

```powershell
python .\scripts\evaluate_demo.py `
  --demo-dir .\outputs\voiceprint_demo_ecapa_multi\meeting `
  --reference .\config\speaker_corrections.demo.json `
  --output-dir .\outputs\voiceprint_demo_ecapa_multi\meeting\figures

python .\scripts\evaluate_demo.py `
  --demo-dir .\outputs\voiceprint_demo_ecapa_multi\meeting2 `
  --reference .\config\speaker_corrections.meeting2.json `
  --output-dir .\outputs\voiceprint_demo_ecapa_multi\meeting2\figures
```

也可以生成两个会议合并后的总评估图表：

```powershell
python .\scripts\evaluate_multi_meeting_demo.py `
  --multi-dir .\outputs\voiceprint_demo_ecapa_multi `
  --output-dir .\outputs\voiceprint_demo_ecapa_multi\figures
```

当前已经生成三套评估输出：

| 评估范围 | 输出目录 |
|---|---|
| `meeting.mp3` 单独评估 | `F:\实习\voiceprint_solution_work\phase1_whisperx_stt\outputs\voiceprint_demo_ecapa_multi\meeting\figures` |
| `meeting2.mp3` 单独评估 | `F:\实习\voiceprint_solution_work\phase1_whisperx_stt\outputs\voiceprint_demo_ecapa_multi\meeting2\figures` |
| 两个会议合并评估 | `F:\实习\voiceprint_solution_work\phase1_whisperx_stt\outputs\voiceprint_demo_ecapa_multi\figures` |

## 6. 当前 ECAPA 版结果

本次 ECAPA 版 demo 已跑通，关键结果如下：

| 指标 | 数值 |
|---|---:|
| WhisperX 原始转写 CER | 0.4225 |
| WhisperX 原始转写 WER | 0.4232 |
| 校正后转写 CER | 0.0000 |
| 校正后转写 WER | 0.0000 |
| 自动声纹窗口准确率（排除 GROUP） | 0.9375 |
| 自动声纹宏平均召回率 | 0.9286 |
| 自动声纹微平均召回率（排除 GROUP） | 0.9375 |
| WhisperX RTF | 0.1602 |

各员工召回率：

| 说话人 | 正确窗口 | 总窗口 | 召回率 |
|---|---:|---:|---:|
| A | 27 | 27 | 1.0000 |
| B | 13 | 14 | 0.9286 |
| C | 11 | 14 | 0.7857 |
| D | 9 | 9 | 1.0000 |

相比 MFCC 兜底版，窗口准确率从 `0.6875` 提升到 `0.9375`。

## 7. 输出目录

ECAPA 版输出目录：

```text
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\outputs\voiceprint_demo_ecapa
```

核心输出：

| 文件 | 说明 |
|---|---|
| `meeting.transcript.json` | WhisperX 原始转写 JSON |
| `meeting.transcript.txt` | WhisperX 原始转写文本 |
| `meeting.transcript.srt` | WhisperX 原始字幕 |
| `meeting.corrected_transcript.json` | 使用参考正文校正后的转写 JSON |
| `meeting.corrected_transcript.txt` | 校正后纯文本 |
| `meeting.corrected_transcript.srt` | 校正后字幕 |
| `voiceprint_profiles.json` | A/B/C/D 注册声纹库摘要，包含 ECAPA 后端、模型路径、embedding 维度 |
| `speaker_windows.csv` | 每个滑动窗口的 A/B/C/D 余弦相似度和预测身份 |
| `final_demo_result.json` | 最终结构化结果 |
| `meeting_transcript_with_speakers.md` | 带说话人标签的可读会议记录 |
| `demo_summary.md` | Demo 摘要 |

评估图表：

```text
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\outputs\voiceprint_demo_ecapa\figures
```

| 文件 | 说明 |
|---|---|
| `01_stt_cer_wer.png` | WhisperX 原始转写与校正后转写的 CER/WER 对比 |
| `02_voiceprint_similarity_curves.png` | A/B/C/D 声纹相似度随时间变化 |
| `03_voiceprint_confidence_curve.png` | 声纹识别置信度随时间变化 |
| `04_speaker_confusion_matrix.png` | 自动声纹匹配与参考标注的窗口级混淆矩阵 |
| `05_speaker_duration_bar.png` | 校正后各说话人发言时长 |
| `06_runtime_rtf_bar.png` | 音频时长、推理耗时和 RTF |
| `evaluation_report.md` | 图表说明与关键指标 |
| `evaluation_metrics.json` | 机器可读评估指标 |

## 8. 与实习任务的对应关系

| 实习任务要求 | 当前 demo 对应实现 |
|---|---|
| WhisperX 预训练大模型部署、评测与推理加速 | 已配置 WhisperX/faster-whisper tiny，本地 CUDA 推理，输出 CER/WER 和 RTF |
| Speaker Diarization / 声纹聚类提取底层声学特征 | 当前用滑动窗口 + ECAPA speaker embedding 做员工声纹匹配；后续可接 pyannote diarization 做更精细切分 |
| 构建员工声纹库 | 使用 A/B/C/D 单人注册录音提取 ECAPA embedding，生成 `voiceprint_profiles.json` |
| LLM 语义理解完成身份对齐与映射 | 当前先用人工校正文件模拟参考标注；后续可将 `workflow_payload` 交给 LLM/Agent 做语义校验、摘要和待办抽取 |
| Agent Workflow 端到端结构化打通 | 已输出 `final_demo_result.json` 和 `workflow_payload`，可直接供上游 Agent 工作流读取 |

## 9. 当前边界与下一步

当前 demo 已能展示“注册声纹库 + 会议身份识别”的闭环，但还不是完整生产系统。

下一步建议：

1. 将 WhisperX 模型从 tiny 升级到 small/medium，提高中文转写质量。
2. 引入 pyannote diarization，先做说话人切分，再做员工身份映射。
3. 为每位员工录制更多注册音频，例如 3 段、每段 20 到 30 秒，覆盖不同语速和麦克风距离。
4. 加入阈值策略：低相似度或低 margin 的片段标记为 `UNKNOWN` 或 `needs_review`。
5. 接入 LLM/Agent：基于结构化片段生成会议摘要、任务、负责人和截止时间。
