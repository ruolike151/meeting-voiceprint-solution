# 第一阶段 WhisperX STT 运行环境配置记录

## 1. 配置结果

已在本机完成第一阶段 WhisperX STT 运行环境配置，环境目录为：

```text
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\.venv
```

当前环境支持：

- WhisperX 语音转文本推理。
- PyTorch GPU 推理。
- CUDA 版 `torch / torchaudio / torchvision`。
- 项目内 ffmpeg 兜底，不依赖系统 PATH。
- 项目内模型缓存，避免写入无权限的用户缓存目录。
- 本地 faster-whisper tiny 模型冒烟测试。

## 2. 已安装核心版本

| 组件 | 版本 / 状态 |
|---|---|
| Python | 3.12.9 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| NVIDIA Driver | 591.74 |
| Driver CUDA Capability | 13.1 |
| PyTorch | 2.8.0+cu128 |
| Torchaudio | 2.8.0+cu128 |
| Torchvision | 0.23.0+cu128 |
| WhisperX | 3.8.6 |
| faster-whisper | 1.2.1 |
| ctranslate2 | 4.8.1 |
| imageio-ffmpeg | 0.6.0 |
| jiwer | 4.0.0 |

## 3. 激活环境

```powershell
cd F:\实习\voiceprint_solution_work\phase1_whisperx_stt
.\.venv\Scripts\Activate.ps1
```

也可以不激活环境，直接使用：

```powershell
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\.venv\Scripts\python.exe
```

## 4. 环境检查

运行：

```powershell
python .\scripts\env_check.py
```

应能看到：

```text
cuda_available: True
cuda_version: 12.8
gpu: NVIDIA GeForce RTX 4060 Laptop GPU
```

## 5. 已完成冒烟测试

测试音频：

```text
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\samples\smoke_en.wav
```

使用本地 tiny 模型：

```text
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\models\faster-whisper-tiny
```

运行命令：

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

输出文件：

```text
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\outputs\smoke_test\smoke_en.transcript.json
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\outputs\smoke_test\smoke_en.transcript.txt
F:\实习\voiceprint_solution_work\phase1_whisperx_stt\outputs\smoke_test\smoke_en.transcript.srt
```

识别结果：

```text
Hello Team, this is A Whisper X Environment Test.
```

本次测试耗时：

```text
14.202 seconds
```

说明：首次运行包含模型初始化、VAD 初始化和 Matplotlib 字体缓存构建，后续同模型运行会更快。

## 6. 模型下载说明

自动从 Hugging Face 下载模型时，本机网络环境出现过卡住现象。因此当前已手动将 `Systran/faster-whisper-tiny` 下载到项目本地：

```text
models\faster-whisper-tiny
```

后续测试更大模型时，建议优先使用本地模型目录，或通过镜像按文件下载：

```text
https://hf-mirror.com/Systran/faster-whisper-模型名/resolve/main/文件名
```

## 7. 已处理的问题

| 问题 | 处理方式 |
|---|---|
| 系统未安装 ffmpeg | 使用 `imageio-ffmpeg` 并自动生成项目内 `.bin\ffmpeg.exe` |
| Hugging Face 自动下载卡住 | 改用本地模型目录 |
| WhisperX VAD 写用户缓存无权限 | 默认使用项目内 `.cache` |
| pip 可能把 GPU 版 torch 换成 CPU 版 | 在 `requirements.txt` 中固定 CUDA 12.8 版本 |

## 8. 当前注意事项

- 运行时出现 `torchcodec is not installed correctly` 警告，但第一阶段 STT 已成功完成推理输出；该警告来自 pyannote 的内置音频解码路径，当前链路使用 WhisperX/ffmpeg 解码，不影响转写。
- 使用 `medium` 或 `large` 模型前，需要先下载对应 faster-whisper 模型文件。
- 如果后续进入 Speaker Diarization，可能需要 Hugging Face token 访问 pyannote 相关模型。
