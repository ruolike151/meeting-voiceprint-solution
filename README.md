# 会议语音解析与员工声纹识别系统

基于 **WhisperX** 语音转写与 **SpeechBrain ECAPA-TDNN** 声纹识别的会议语音解析模块：输入一段多人会议录音，自动输出「谁、在什么时间、说了什么」，并以结构化 JSON 的形式交付给下游 Agent 工作流。

> 本项目为实习期间完成的工程实践项目，实现了从**语音转写 → 说话人身份映射 → 结构化数据贯通**的端到端流程。

---

## 一、项目简介

多人会议场景下，录音整理成本高、责任归属难确认。本项目将会议录音解析为**带说话人身份和时间戳的结构化文本**，为下游的会议纪要、待办提取和 Agent 工作流提供可直接消费的数据。

**输入**：一段多人会议录音 + 每位员工的一段单独注册录音

**输出**：结构化 JSON（说话人时间轴、带身份的文本片段、会议全文），供 Agent Workflow 直接读取

## 二、核心能力

- **语音转文字（STT）**：基于 WhisperX 完成会议录音转写，支持中英混合与自动语种识别，输出带时间戳的 JSON / TXT / SRT。
- **音频标准化**：使用 ffmpeg 统一为 16kHz、单声道、16-bit PCM，保证输入一致。
- **员工声纹注册**：每位员工录制一段注册语音，用 ECAPA-TDNN 提取 192 维声纹向量，构建可动态维护的员工声纹库。
- **说话人身份映射**：会议音频按滑动窗口切分，逐窗提取声纹向量并与员工声纹库做余弦相似度匹配，判定「这段是谁在说」。
- **文本与身份对齐**：将转写文本按时间戳与说话人结果绑定，输出可读的会议记录。
- **结构化数据贯通**：输出 `final_demo_result.json` 与 `workflow_payload`，与下游工作流解耦。
- **量化评测**：输出 CER/WER、声纹相似度曲线、混淆矩阵、各说话人召回率、RTF 等指标与图表。
- **动态增删员工**：新增员工只需增加注册音频与声纹向量，删除员工只需移除对应 profile，**无需重新训练模型**。

## 三、系统流程

```text
多人会议录音
   │
   ├─ ① 音频预处理：ffmpeg → 16kHz / 单声道 / 16-bit PCM
   │
   ├─ ② 语音转写（WhisperX）
   │      → 带时间戳的文本（JSON / TXT / SRT）
   │
   ├─ ③ 声纹识别（SpeechBrain ECAPA-TDNN）
   │      员工注册录音 → 192 维声纹向量 → 员工声纹库
   │      会议音频滑动窗口 → 逐窗提取向量 → 余弦相似度匹配
   │
   ├─ ④ 对齐：转写文本 × 说话人身份（按时间戳绑定）
   │
   └─ ⑤ 结构化输出：final_demo_result.json / workflow_payload
          → Agent Workflow（会议纪要、待办、负责人提取）
```

## 四、技术栈

| 环节 | 使用技术 | 说明 |
| --- | --- | --- |
| 编程语言 | Python 3.12 | 主开发语言 |
| 语音转写 | WhisperX（faster-whisper） | 开源语音识别，输出词级时间戳 |
| 声纹识别 | SpeechBrain ECAPA-TDNN（`speechbrain/spkrec-ecapa-voxceleb`） | 预训练说话人 embedding，192 维 |
| 身份匹配 | 余弦相似度 + 滑动窗口 | 可解释、无需完整说话人分离 |
| 深度学习框架 | PyTorch + CUDA | GPU 推理，支持 float16 加速 |
| 音频处理 | ffmpeg / imageio-ffmpeg / soundfile / torchaudio | 解码、重采样、标准化 |
| 评测与图表 | CER / WER / 混淆矩阵 / RTF + matplotlib | 量化评估识别效果与性能 |
| 输出格式 | JSON / CSV / Markdown / SRT | 便于下游消费 |

> 说明：语音转写与声纹识别使用的是 **WhisperX、ECAPA-TDNN 这类专用语音模型**，并非通用大语言模型；大语言模型（LLM）仅作为后续会议摘要能力的可选增强方向。

## 五、目录结构

```text
voiceprint_solution_work/
├── README.md                      # 本文件
├── 项目目录说明.md                 # 项目结构说明
├── build_voiceprint_solution.py    # 生成总体方案 Word 文档的脚本
├── convert_docx_to_markdown.py     # Word 方案转 Markdown 脚本
├── architecture.png                # 系统总体架构图
├── employee_lifecycle.png          # 员工声纹生命周期图
├── deliverables/                   # 交付文档
│   ├── 员工声纹识别系统详细方案.md / .docx
│   ├── 第一阶段_WhisperX_STT部署与评测文档.md
│   ├── 第一阶段_WhisperX_STT运行环境配置记录.md
│   └── 小Demo_四人会议声纹识别运行说明.md
├── phase1_whisperx_stt/            # 第一阶段可运行代码
│   ├── README.md
│   ├── requirements.txt
│   ├── config.example.json
│   ├── src/whisperx_stt/           # 音频处理与转写模块
│   └── scripts/                    # 转写 / 评测 / 声纹 demo 入口
└── ppt_tasks/defense_ppt/          # 答辩演示材料与构建脚本
```

## 六、快速开始

### 1. 环境准备

```powershell
cd phase1_whisperx_stt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

检查运行环境（Python、PyTorch、CUDA、ffmpeg 等）：

```powershell
python .\scripts\env_check.py
```

### 2. 准备数据

本项目**不包含**任何真实录音数据，请在项目根目录自行创建 `data/` 目录并放入：

```text
data/
├── meeting.mp3     # 多人会议录音
├── A_solo.mp3      # 员工 A 的注册录音
├── B_solo.mp3      # 员工 B 的注册录音
├── C_solo.mp3      # 员工 C 的注册录音
└── D_solo.mp3      # 员工 D 的注册录音
```

### 3. 运行语音转写

```powershell
python .\scripts\transcribe.py `
  --input .\samples\smoke_en.wav `
  --output-dir .\outputs\smoke_test `
  --model .\models\faster-whisper-tiny `
  --device cuda `
  --compute-type float16 `
  --language en `
  --batch-size 4 `
  --no-align
```

### 4. 运行四人会议声纹识别 Demo

```powershell
python .\scripts\run_voiceprint_demo.py `
  --data-dir ..\data `
  --output-dir .\outputs\voiceprint_demo `
  --model .\models\faster-whisper-tiny `
  --device cuda `
  --compute-type float16 `
  --language auto `
  --batch-size 8
```

使用预训练 ECAPA-TDNN 声纹后端：

```powershell
python .\scripts\run_voiceprint_demo.py `
  --data-dir ..\data `
  --output-dir .\outputs\voiceprint_demo_ecapa `
  --model .\models\faster-whisper-tiny `
  --device cuda `
  --compute-type float16 `
  --language auto `
  --batch-size 8 `
  --embedding-backend speechbrain_ecapa `
  --speaker-model-path .\models\speechbrain-spkrec-ecapa-voxceleb `
  --strict-embedding-backend
```

### 5. 生成评测图表

```powershell
python .\scripts\evaluate_demo.py `
  --demo-dir .\outputs\voiceprint_demo_ecapa `
  --reference .\config\speaker_corrections.demo.json `
  --output-dir .\outputs\voiceprint_demo_ecapa\figures
```

> 参考标注文件（`speaker_corrections.*.json`）包含真实会议文本，未纳入本仓库，需要按同一格式自行准备。

## 七、运行结果

在真实会议录音上的评测结果：

| 指标 | 结果 |
| --- | --- |
| 测试规模 | 2 场会议，共 444.5 秒，355 个声纹窗口 |
| 自动声纹窗口准确率（排除多人合说） | **93.75%** |
| 宏平均召回率 | **92.86%** |
| 各说话人召回率 | A 100% / B 92.9% / C 78.6% / D 100% |
| 推理实时率 RTF | **0.16**（约 6 倍实时） |
| MFCC 兜底特征基线（对比） | 68.75% |

**关键结论**：声纹后端从轻量 MFCC 统计特征升级为 ECAPA-TDNN 预训练声纹向量后，窗口准确率由 68.75% 提升至 93.75%，身份识别效果显著改善。

## 八、数据与隐私说明

本仓库已通过 `.gitignore` 排除以下内容，**不会上传任何真实语音数据与会议文本**：

- 真实会议录音与员工声纹注册音频（含生物特征，属于敏感个人信息）；
- 会议转写文本、人工校正配置与识别输出结果；
- 本地虚拟环境、模型权重与缓存等大文件。

如需完整复现，请使用**经授权、且已脱敏**的音频数据，并注意遵守相关法律法规与单位内部的数据安全规定。声纹属于个人敏感生物特征信息，禁止用于未授权场景。

## 九、已知限制

- 语音转写使用 faster-whisper tiny 模型，中文转写质量有限，可通过更换 small / medium / large 模型提升。
- 身份匹配采用滑动窗口而非完整说话人分离，多人同时说话时只能识别主导声纹，多人合说片段未做拆分。
- 会议正文与人物关系使用了人工校正配置，用于保证展示结果正确；自动声纹结果仍保留在窗口明细与混淆矩阵中用于客观评估。
- 大语言模型部分目前仅输出可消费的结构化数据，尚未接入真实 LLM API。
- 注册音频时长较短、录音环境差异较大时，声纹识别稳定性会下降。

## 十、后续计划

| 优先级 | 工作 | 目的 |
| --- | --- | --- |
| P0 | 转写模型升级为 small / medium | 提升中文转写质量 |
| P0 | 引入 pyannote 说话人分离 | 更准确地切分说话人边界 |
| P1 | 每位员工补充多段注册音频 | 提升声纹库鲁棒性 |
| P1 | 设置 unknown / needs_review 阈值 | 降低误识别风险 |
| P1 | 接入 LLM | 输出会议摘要、待办事项与负责人 |
| P2 | 扩大测试集 | 统计 CER、WER、DER、召回率等长期指标 |

## 十一、说明

本项目为实习期间完成的工程实践成果，包含可运行代码、技术方案文档与答辩演示材料。文档中出现的员工身份均以 A / B / C / D 代称，不涉及真实姓名。
