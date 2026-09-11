# 小 Demo：四人会议语音转写与员工声纹识别运行说明

## 1. Demo 目标

本 demo 对齐当前实习任务要求，完成一条可运行的端到端流程：

```text
四人会议录音
  -> WhisperX 语音转文本（STT）
  -> A/B/C/D 单人注册录音构建员工声纹库
  -> SpeechBrain ECAPA-TDNN 预训练声纹 embedding
  -> 会议滑动窗口身份匹配
  -> 说话人身份与转写文本对齐
  -> 输出 Agent Workflow 可读取的结构化 JSON
```

本次已按推荐路线完成修改：声纹识别后端从原来的轻量 MFCC 统计特征升级为 `SpeechBrain ECAPA-TDNN` 预训练 speaker embedding。MFCC 仍保留为兜底方案，用于预训练模型缺失或加载失败时保证 demo 可运行。

## 2. 当前完成的工作

| 模块 | 完成情况 |
|---|---|
| WhisperX STT | 已能对 `meeting.mp3` 完成语音转文本，并输出 JSON/TXT/SRT |
| 员工声纹注册 | 已使用 `A_solo.mp3`、`B_solo.mp3`、`C_solo.mp3`、`D_solo.mp3` 构建声纹库 |
| 预训练声纹模型 | 已接入 `speechbrain/spkrec-ecapa-voxceleb`，本地模型目录约 84.86 MB |
| 声纹匹配 | 已按滑动窗口提取会议音频 ECAPA embedding，并与员工声纹库做余弦相似度匹配 |
| 动态员工库思路 | 新增员工只需添加注册音频并重建该员工 embedding；删除员工只需从声纹库移除对应 profile |
| 评估图表 | 已输出 CER/WER、相似度曲线、置信度曲线、混淆矩阵、发言时长、RTF |
| 结构化打通 | 已输出 `final_demo_result.json` 和 `workflow_payload`，可供后续 Agent Workflow 使用 |

## 3. 项目路径

主项目目录：

```text
F:\实习\voiceprint_solution_work
```

Demo 工作目录：

```text
F:\实习\voiceprint_solution_work\phase1_whisperx_stt
```

虚拟环境：

```text
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\.venv
```

数据目录：

```text
F:\实习\voiceprint_solution_work\data
```

ECAPA 版输出目录：

```text
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\outputs\voiceprint_demo_ecapa
```

## 4. 输入数据

当前 demo 使用以下 5 个音频文件：

| 文件 | 用途 | 当前检测时长 |
|---|---|---:|
| `meeting.mp3` | 四人会议完整录音 | 86.52 秒 |
| `A_solo.mp3` | 员工 A 单独注册录音 | 22.416 秒 |
| `B_solo.mp3` | 员工 B 单独注册录音 | 17.760 秒 |
| `C_solo.mp3` | 员工 C 单独注册录音 | 21.720 秒 |
| `D_solo.mp3` | 员工 D 单独注册录音 | 18.000 秒 |

数据含义：

- `meeting.mp3` 是一整个会议录音，用来模拟真实会议场景。
- `A_solo.mp3` 到 `D_solo.mp3` 是每位员工的注册录音，用来建立员工声纹库。
- 当前 15 秒以上的单人注册录音已经可以支撑小 demo。
- 生产建议每位员工准备 30 到 60 秒以上注册音频，最好包含多段、不同语速、不同音量距离的数据。

## 5. 虚拟环境使用方式

在 VS Code 终端中进入项目并激活虚拟环境：

```powershell
cd F:\实习\voiceprint_solution_work\phase1_whisperx_stt
.\.venv\Scripts\Activate.ps1
```

激活成功后，终端前面一般会出现：

```text
(.venv)
```

检查运行环境：

```powershell
python .\scripts\env_check.py
```

## 6. 技术栈

| 类型 | 技术 |
|---|---|
| 编程语言 | Python 3.12 |
| STT 模型 | WhisperX 3.8.6 / faster-whisper tiny |
| 深度学习框架 | PyTorch 2.8.0 + CUDA 12.8 |
| 音频处理 | ffmpeg / imageio-ffmpeg / soundfile / torchaudio |
| 声纹模型 | SpeechBrain ECAPA-TDNN：`speechbrain/spkrec-ecapa-voxceleb` |
| 声纹匹配 | ECAPA embedding + cosine similarity |
| 评估 | CER、WER、窗口准确率、召回率、混淆矩阵、RTF |
| 图表 | matplotlib |
| 结构化输出 | JSON / CSV / Markdown / SRT |

## 7. ECAPA 声纹模型信息

本次推荐路线使用的预训练声纹模型：

```text
speechbrain/spkrec-ecapa-voxceleb
```

本地目录：

```text
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\models\speechbrain-spkrec-ecapa-voxceleb
```

核心文件：

| 文件 | 说明 |
|---|---|
| `hyperparams.yaml` | 模型结构与加载配置 |
| `embedding_model.ckpt` | ECAPA-TDNN 主权重，约 79.46 MiB |
| `classifier.ckpt` | VoxCeleb 分类头权重 |
| `mean_var_norm_emb.ckpt` | embedding 归一化参数 |
| `label_encoder.txt` | 训练说话人标签 |

当前 `voiceprint_profiles.json` 中记录的模型目录大小约为 `84.86 MB`，员工 embedding 维度为 `192`。

## 8. 运行命令

推荐运行 ECAPA 版：

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

参数说明：

| 参数 | 作用 |
|---|---|
| `--data-dir ..\data` | 指定会议录音和注册录音目录 |
| `--output-dir .\outputs\voiceprint_demo_ecapa` | 指定 ECAPA 版输出目录 |
| `--model .\models\faster-whisper-tiny` | 使用本地 faster-whisper tiny 模型 |
| `--device cuda` | 使用 GPU 推理 |
| `--compute-type float16` | 使用 FP16 加速 |
| `--reuse-transcript` | 复用已有 WhisperX 转写，避免重复 STT |
| `--corrections` | 使用会议正文校正文件，保证 demo 展示文本和人物关系正确 |
| `--embedding-backend speechbrain_ecapa` | 使用 ECAPA 预训练声纹 embedding |
| `--strict-embedding-backend` | 严格使用 ECAPA，加载失败时直接报错，避免误用兜底结果 |

如果只是想保证流程能跑通，也可以使用自动模式：

```powershell
python .\scripts\run_voiceprint_demo.py --embedding-backend auto
```

自动模式会优先加载 ECAPA。如果模型缺失、权重不完整或加载失败，会回退到 MFCC 兜底后端，并在输出中记录回退原因。

## 9. 生成评估图表

```powershell
python .\scripts\evaluate_demo.py `
  --demo-dir .\outputs\voiceprint_demo_ecapa `
  --reference .\config\speaker_corrections.demo.json `
  --output-dir .\outputs\voiceprint_demo_ecapa\figures
```

## 10. 输出文件

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
| `meeting.corrected_transcript.json` | 简体中文校正版转写 JSON |
| `meeting.corrected_transcript.txt` | 简体中文校正版纯文本 |
| `meeting.corrected_transcript.srt` | 简体中文校正版字幕 |
| `voiceprint_profiles.json` | 员工声纹库摘要，包含模型后端、模型路径、embedding 数量与维度 |
| `speaker_windows.csv` | 每个会议窗口的 A/B/C/D 相似度、预测说话人、置信度 |
| `final_demo_result.json` | 最终结构化结果 |
| `meeting_transcript_with_speakers.md` | 带说话人标签的可读会议记录 |
| `demo_summary.md` | Demo 摘要 |

图表目录：

```text
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\outputs\voiceprint_demo_ecapa\figures
```

图表文件：

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

## 11. 两个会议一起测试

现在数据目录中包含两段会议音频：

```text
F:\实习\voiceprint_solution_work\data\meeting.mp3
F:\实习\voiceprint_solution_work\data\meeting2.mp3
```

可以使用批量脚本一次性测试两段会议：

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

双会议输出目录：

```text
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\outputs\voiceprint_demo_ecapa_multi
```

新增输出文件：

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

合计：

| 指标 | 数值 |
|---|---:|
| 会议数量 | 2 |
| 总音频时长 | 444.504 秒 |
| 总声纹窗口数 | 355 |

说明：`meeting.mp3` 使用 `speaker_corrections.demo.json` 作为正文标注；`meeting2.mp3` 使用 `speaker_corrections.meeting2.json` 作为正文标注。两个会议的最终展示文本和人物标签都已应用人工校正。

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

## 12. 本次 ECAPA 版运行结果

运行输出：

```text
Speaker embedding backend: speechbrain_ecapa
Transcript segments: 7
Speaker windows: 69
Speaker turns: 7
```

关键指标：

| 指标 | 当前结果 |
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

对比原 MFCC 兜底版：

| 声纹后端 | 窗口准确率（排除 GROUP） | 宏平均召回率 |
|---|---:|---:|
| MFCC 兜底特征 | 0.6875 | 0.7447 |
| SpeechBrain ECAPA-TDNN | 0.9375 | 0.9286 |

可以看到，接入预训练声纹 embedding 后，小 demo 的身份识别效果明显提升。

## 13. 校正后人物关系

本 demo 使用用户提供的会议正文作为参考标注。当前两个会议分别使用以下校正文件：

| 会议 | 校正文件 |
|---|---|
| `meeting.mp3` | `F:\实习\voiceprint_solution_work\phase1_whisperx_stt\config\speaker_corrections.demo.json` |
| `meeting2.mp3` | `F:\实习\voiceprint_solution_work\phase1_whisperx_stt\config\speaker_corrections.meeting2.json` |

`meeting.mp3` 人物对应关系：

| 人物 | 角色 |
|---|---|
| A | 项目经理 |
| B | 运营专员 |
| C | UI 设计师 |
| D | 前端开发 |
| GROUP | B/C/D 合说 |

`meeting2.mp3` 人物对应关系：

| 人物 | 角色 |
|---|---|
| A | 项目负责人 |
| B | 市场专员 |
| C | 设计专员 |
| D | 执行专员 |
| GROUP | B/C/D 合说 |

`meeting.mp3` 校正后发言时长：

| 说话人 | 发言时长 |
|---|---:|
| A（项目经理） | 34.57 秒 |
| B（运营专员） | 16.56 秒 |
| C（UI 设计师） | 16.84 秒 |
| D（前端开发） | 12.41 秒 |
| GROUP（B/C/D 合说） | 3.60 秒 |

说明：校正文件用于保证最终展示文本和人物关系正确；自动声纹窗口结果仍保留在 `speaker_windows.csv` 和混淆矩阵中，用于客观评估模型效果。

## 14. `final_demo_result.json` 结构

最终结构化结果包含以下核心字段：

| 字段 | 作用 |
|---|---|
| `demo_name` | Demo 名称 |
| `meeting_audio` | 原始会议录音路径 |
| `embedding_backend` | 当前声纹后端，ECAPA/MFCC 及模型路径 |
| `speaker_profiles` | A/B/C/D 员工声纹库摘要 |
| `speaker_turns` | 会议中的说话人时间轴 |
| `transcript_segments` | 文本、时间戳、说话人身份绑定后的片段 |
| `full_text` | 拼接后的完整会议正文 |
| `workflow_payload` | 后续 Agent Workflow 可读取的数据结构 |

`workflow_payload` 示例结构：

```json
{
  "meeting_id": "demo_meeting_001",
  "participants": ["A", "B", "C", "D"],
  "segments": [
    {
      "id": 0,
      "start": 0.031,
      "end": 9.8,
      "speaker": "A",
      "speaker_confidence": 1.0,
      "speaker_status": "corrected",
      "text": "各位早，今天开一个三分钟短会..."
    }
  ]
}
```

## 15. 后续如何动态增加或删除员工

### 新增员工

1. 让新员工录制一段单独注册录音，例如 `E_solo.mp3`。
2. 将录音统一转为 16kHz 单声道 WAV。
3. 使用 ECAPA 模型提取该员工的 embedding。
4. 将新员工 profile 加入声纹库。
5. 后续会议匹配时，把 E 加入候选说话人集合。

不需要重新训练 ECAPA 模型。

### 删除员工

1. 从声纹库中删除该员工的 profile。
2. 后续会议匹配时不再把该员工作为候选人。

也不需要重新训练模型。

这也是使用预训练 speaker embedding 的核心优势：员工库变化主要是“增删向量”，不是“重新训练模型”。

## 16. 当前边界

当前 demo 已经能展示实习任务的主要链路，但仍有一些边界：

- WhisperX tiny 中文转写质量有限，所以原始 CER/WER 较高；后续可换 small/medium/large。
- 当前会议切分使用滑动窗口，不是完整 diarization；多人同时说话时只能识别主导声纹。
- GROUP 片段表示多人合说，当前不计入单人准确率。
- 当前 LLM 部分还没有接真实 API，只是先输出了 Agent 可消费的结构化 payload。
- 注册录音越短、环境差异越大，声纹识别越容易不稳定。

## 17. 下一步建议

建议按以下顺序继续做：

| 优先级 | 工作 | 目的 |
|---|---|---|
| P0 | WhisperX tiny 升级为 small/medium | 提升中文转写质量 |
| P0 | 引入 pyannote diarization | 更准确切分说话人边界 |
| P1 | 每位员工增加多段注册音频 | 提升声纹库鲁棒性 |
| P1 | 设置 `UNKNOWN/needs_review` 阈值 | 降低误识别风险 |
| P1 | 接入 LLM/Agent | 输出摘要、待办事项、负责人和时间线 |
| P2 | 增加更大测试集 | 统计 CER、WER、DER、召回率、准确率等长期指标 |
