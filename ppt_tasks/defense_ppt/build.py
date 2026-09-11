# -*- coding: utf-8 -*-
"""Build the internship defense deck (minimal engineering direction)."""
import hashlib, json
from pathlib import Path
from pptx_designer import Presentation, validate_resolved_theme
from pptx_designer.renderer.theme import ThemeComposer
from pptx_designer.tools.layout import add_slide, page_header, page_number, top_bar
from pptx_designer.tools.text import multiline, text
from pptx_designer.tools.shapes import arrow, rect, rrect
from pptx_designer.tools.cards import kpi_card
from pptx_designer.tools.charts import bar_chart

TASK_DIR = Path(r"F:\实习\voiceprint_solution_work\ppt_tasks\defense_ppt")
OUT_DIR = TASK_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TOTAL = 9

def make_theme():
    theme = ThemeComposer().compose(style="minimalism-swiss-style", seed=7)
    theme["typography"] = {"heading": "Microsoft YaHei", "body": "Microsoft YaHei"}
    validate_resolved_theme(theme)
    fp = hashlib.sha256(json.dumps(theme, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    (TASK_DIR / "resolved-theme-v1.json").write_text(json.dumps(theme, ensure_ascii=False, indent=2), encoding="utf-8")
    return theme, fp

def card(slide, x, y, w, h, accent_stripe=True):
    rrect(slide, x, y, w, h, "surface", line="border")
    if accent_stripe:
        rect(slide, x, y, 0.07, h, "accent")

def build(theme):
    prs = Presentation(theme=theme, strict_theme=True)
    s = add_slide(prs)
    top_bar(s, "accent")
    text(s, 0.9, 2.35, 11.6, 1.6, "会议语音解析与员工声纹识别系统", font_size=42, color="ink", bold=True)
    rect(s, 0.9, 3.95, 1.8, 0.05, "accent")
    text(s, 0.9, 4.18, 11.6, 0.6, "基于 WhisperX 语音转写与 ECAPA 声纹识别", font_size=20, color="accent")
    text(s, 0.9, 6.05, 11.6, 0.5, "汇报人：（姓名）      实习单位：（学校/学院）      时间：2026 年 9 月", font_size=12, color="muted")
    rrect(s, 10.7, 0.5, 1.9, 0.55, "surface", line="border")
    text(s, 10.7, 0.58, 1.9, 0.4, "实习答辩", font_size=12, color="accent", align="center")

    s = add_slide(prs)
    top_bar(s, "accent")
    page_header(s, "实习任务与目标", "从需求理解到工程交付，完成会议语音解析模块")
    text(s, 0.7, 1.5, 11.9, 0.5, "目标：按期交付稳定运行的语音解析模块源码，与下游工作流数据贯通，并提交完整开发文档。", font_size=13, color="ink")
    tasks = [("01", "学习 Agent 开发流程", "掌握 vibe coding 与 AI 编程协作方式，理解企业级开发规范"), ("02", "掌握 AI 语音模型落地", "学会 WhisperX 语音大模型的部署、推理与工程化方法"), ("03", "完成交付与数据贯通", "交付可运行模块，与下游工作流数据贯通并提交文档")]
    xs = [0.7, 4.83, 8.96]
    for (num, title, body), x in zip(tasks, xs):
        card(s, x, 2.0, 3.68, 2.7)
        text(s, x + 0.28, 2.22, 0.9, 0.5, num, font_size=26, color="accent", bold=True)
        text(s, x + 0.28, 2.78, 3.2, 0.5, title, font_size=17, color="ink", bold=True)
        multiline(s, x + 0.28, 3.35, 3.15, 1.2, [body], font_size=12, color="muted")
    rrect(s, 0.7, 5.05, 11.93, 0.85, "surface", line="border")
    rect(s, 0.7, 5.05, 0.07, 0.85, "accent")
    text(s, 1.05, 5.25, 11.4, 0.5, "交付物：可运行源码    ·    结构化 JSON 数据接口    ·    完整开发文档", font_size=14, color="ink")
    page_number(s, 2, TOTAL)

    s = add_slide(prs)
    top_bar(s, "accent")
    page_header(s, "项目整体流程", "一段会议录音 → 谁、在什么时间、说了什么")
    stages = [("01", "语音转写（STT）", ["音频预处理（16kHz·单声道）", "WhisperX 语音转写", "输出带时间戳文本"]), ("02", "声纹识别", ["注册录音→ECAPA 声纹库", "滑动窗口提取 192 维向量", "余弦相似度匹配说话人"]), ("03", "数据贯通", ["文本 × 说话人对齐", "输出结构化 JSON", "供 Agent Workflow 消费"])]
    cx = [0.7, 4.89, 9.08]
    for (num, title, steps), x in zip(stages, cx):
        card(s, x, 1.95, 3.55, 3.45)
        text(s, x + 0.28, 2.15, 0.9, 0.55, num, font_size=26, color="accent", bold=True)
        text(s, x + 0.28, 2.75, 3.0, 0.5, title, font_size=18, color="ink", bold=True)
        y = 3.42
        for idx, step in enumerate(steps):
            text(s, x + 0.32, y, 2.95, 0.5, step, font_size=12, color="ink")
            if idx < len(steps) - 1:
                arrow(s, x + 1.5, y + 0.5, 0.34, 0.16, "accent")
            y += 0.66
    arrow(s, 4.27, 3.35, 0.58, 0.4, "accent")
    arrow(s, 8.46, 3.35, 0.58, 0.4, "accent")
    text(s, 0.7, 5.5, 11.9, 0.75, "转写：WhisperX（faster-whisper）    ·    声纹：SpeechBrain ECAPA-TDNN（192 维）    ·    下游：Agent Workflow", font_size=11, color="muted")
    page_number(s, 3, TOTAL)

    s = add_slide(prs)
    top_bar(s, "accent")
    page_header(s, "技术选型与使用理由", "为什么这样选：可落地、可扩展、可复用")
    tech = [("WhisperX", "语音转写 + 词级时间戳", "开源、支持中英混合；时间戳是说话人对齐和数据贯通的基础"), ("SpeechBrain ECAPA-TDNN", "提取 192 维声纹向量", "预训练即用、免训练；员工动态增删只需增删向量，无需重训模型"), ("余弦相似度 + 滑动窗口", "会议逐段身份匹配", "可解释、稳定；不依赖完整说话人分离，demo 即可跑通"), ("PyTorch / CUDA + ffmpeg", "GPU 推理 + 音频预处理", "float16 加速推理；统一 16kHz 单声道，降低异常输入影响")]
    pos = [(0.7, 1.9), (6.88, 1.9), (0.7, 4.15), (6.88, 4.15)]
    for (name, role, why), (x, y) in zip(tech, pos):
        card(s, x, y, 5.75, 2.0)
        text(s, x + 0.28, y + 0.2, 5.2, 0.45, name, font_size=17, color="ink", bold=True)
        text(s, x + 0.28, y + 0.74, 5.2, 0.4, "作用：" + role, font_size=12, color="muted")
        multiline(s, x + 0.28, y + 1.14, 5.2, 0.8, ["为什么：" + why], font_size=12, color="ink")
    page_number(s, 4, TOTAL)

    s = add_slide(prs)
    top_bar(s, "accent")
    page_header(s, "声纹识别：方案与效果", "为什么用预训练向量 + 检索，而不是训练一个分类模型")
    text(s, 1.55, 1.7, 4.6, 0.4, "窗口准确率（排除 GROUP）", font_size=14, color="ink", bold=True)
    bar_chart(s, 1.55, 2.15, [("MFCC 特征", 0.6875, "68.8%"), ("ECAPA-TDNN", 0.9375, "93.8%")], max_width=4.3)
    text(s, 1.55, 3.55, 4.6, 0.4, "宏平均召回率", font_size=14, color="ink", bold=True)
    bar_chart(s, 1.55, 4.0, [("MFCC 特征", 0.7447, "74.5%"), ("ECAPA-TDNN", 0.9286, "92.9%")], max_width=4.3)
    text(s, 7.0, 1.7, 5.5, 0.4, "为什么用向量 + 检索", font_size=15, color="ink", bold=True)
    reasons = ["员工动态增删：只需增删向量，模型不改、不用重训", "可拒绝未知人：开集识别，低于阈值返回 unknown 而非强猜", "小样本快落地：预训练模型 + 检索，两人团队 6 周可试点"]
    y = 2.25
    for r in reasons:
        rect(s, 7.0, y + 0.06, 0.16, 0.16, "accent")
        text(s, 7.32, y, 5.1, 0.8, r, font_size=13, color="ink")
        y += 0.85
    text(s, 0.7, 6.4, 11.9, 0.4, "数据来源：双会议自动声纹窗口评测（参考标注），详见项目 demo 文档", font_size=10, color="muted")
    page_number(s, 5, TOTAL)

    s = add_slide(prs)
    top_bar(s, "accent")
    page_header(s, "项目结果", "两场真实会议 · 共 444.5 秒 · 355 个声纹窗口")
    kpis = [("93.75%", "窗口准确率"), ("92.86%", "宏平均召回率"), ("0.16", "RTF（约 6× 实时）"), ("444.5s", "测试规模（2 场会议）")]
    kx = [0.7, 3.73, 6.76, 9.79]
    for (num, label), x in zip(kpis, kx):
        kpi_card(s, x, 1.85, 2.85, 1.5, num, label)
    text(s, 1.55, 3.75, 4.6, 0.4, "各说话人识别召回率", font_size=14, color="ink", bold=True)
    bar_chart(s, 1.55, 4.15, [("A", 1.0, "100%"), ("B", 0.9286, "92.9%"), ("C", 0.7857, "78.6%"), ("D", 1.0, "100%")], max_width=4.3)
    rrect(s, 7.0, 3.9, 5.6, 1.95, "surface", line="border")
    rect(s, 7.0, 3.9, 0.07, 1.95, "accent")
    multiline(s, 7.3, 4.1, 5.1, 1.6, ["说明：", "· A、D 召回率 100%，B 92.9%，C 78.6%", "· 转写模型：faster-whisper tiny", "· 自动声纹结果与人工校正参考标注对比"], font_size=12, color="ink")
    page_number(s, 6, TOTAL)

    s = add_slide(prs)
    top_bar(s, "accent")
    page_header(s, "开发中遇到的问题与解决", "在真实会议测试与性能优化中逐步解决")
    rows = [("环境搭建", "CUDA / PyTorch 版本不匹配、缺 ffmpeg", "虚拟环境固化依赖 + 本地 ffmpeg + 环境记录文档"), ("匹配准确率低", "MFCC 特征区分度不足", "升级 ECAPA-TDNN 预训练向量，窗口准确率 68.8% → 93.8%"), ("真实录音噪声 / 重叠说话", "多人同时说话、环境噪声干扰", "音频降噪 + 滑动窗口 + 人工纠错配置（保留自动结果客观评估）"), ("长音频耗时长", "整段会议转写与特征提取慢", "GPU + FP16 + batch 调优，RTF 降至 0.16")]
    y = 2.0
    for prob, cause, sol in rows:
        rrect(s, 0.7, y, 11.93, 1.0, "surface", line="border")
        rect(s, 6.5, y + 0.12, 0.012, 0.76, "border")
        text(s, 0.95, y + 0.12, 5.3, 0.4, "问题：" + prob, font_size=13, color="ink", bold=True)
        text(s, 0.95, y + 0.52, 5.3, 0.4, cause, font_size=11.5, color="muted")
        text(s, 6.75, y + 0.12, 0.7, 0.4, "解决：", font_size=13, color="accent", bold=True)
        text(s, 7.5, y + 0.12, 4.9, 0.8, sol, font_size=12, color="ink")
        y += 1.18
    page_number(s, 7, TOTAL)

    s = add_slide(prs)
    top_bar(s, "accent")
    page_header(s, "实习总结与后续方向", "收获与迭代计划")
    card(s, 0.7, 1.95, 5.75, 3.5)
    text(s, 0.98, 2.15, 5.2, 0.45, "实习收获", font_size=17, color="ink", bold=True)
    gains = ["完整经历 AI 应用落地全流程：需求→部署→算法→测试→交付", "掌握 Agent / vibe coding 协作方式，提升工程效率", "建立文档、数据贯通、量化评测的工程习惯"]
    y = 2.75
    for g in gains:
        rect(s, 0.98, y + 0.05, 0.16, 0.16, "accent")
        text(s, 1.3, y, 4.9, 0.8, g, font_size=12.5, color="ink")
        y += 0.95
    card(s, 6.88, 1.95, 5.75, 3.5)
    text(s, 7.16, 2.15, 5.2, 0.45, "后续方向", font_size=17, color="ink", bold=True)
    nexts = ["换更大转写模型，提升中文转写质量（CER）", "接入 pyannote 说话人分离，应对多人和说", "接入 LLM，输出会议摘要 / 待办 / 负责人", "多段注册音频，提升声纹库鲁棒性"]
    y = 2.75
    for n in nexts:
        rect(s, 7.16, y + 0.05, 0.16, 0.16, "accent")
        text(s, 7.48, y, 4.9, 0.7, n, font_size=12.5, color="ink")
        y += 0.62
    page_number(s, 8, TOTAL)

    s = add_slide(prs)
    top_bar(s, "accent")
    text(s, 0, 2.6, 13.33, 1.2, "谢谢聆听", font_size=44, color="ink", bold=True, align="center")
    rect(s, 6.0, 3.95, 1.33, 0.05, "accent")
    text(s, 0, 4.2, 13.33, 0.6, "请各位老师批评指正", font_size=18, color="muted", align="center")
    text(s, 0, 6.5, 13.33, 0.4, "实习答辩 · 会议语音解析与员工声纹识别系统", font_size=11, color="muted", align="center")
    return prs

def main():
    theme, fp = make_theme()
    prs = build(theme)
    out = OUT_DIR / "defense_ppt.pptx"
    prs.save(str(out))
    import pptx_designer
    print("SAVED", out)
    print("SLIDES", len(prs.slides))
    print("THEME_FINGERPRINT", fp)
    print("PPTX_DESIGNER_VERSION", getattr(pptx_designer, "__version__", "?"))
    print("PPTX_DESIGNER_PATH", pptx_designer.__file__)

if __name__ == "__main__":
    main()
