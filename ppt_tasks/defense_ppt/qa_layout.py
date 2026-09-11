from pptx import Presentation
from pptx.util import Emu

PATH = r"F:\实习\voiceprint_solution_work\ppt_tasks\defense_ppt\output\defense_ppt.pptx"
W, H = 13.3333, 7.5
prs = Presentation(PATH)

def inch(v):
    return v / 914400.0

def char_w(ch, fs):
    # CJK ~ fs/72 inch, ascii ~ 0.5*fs/72
    pts = fs / 72.0
    return pts if ord(ch) > 0x2E7F else pts * 0.52

def line_h(fs):
    return fs * 1.25 / 72.0

def para_font(p, default=12):
    sizes = [r.font.size.pt for r in p.runs if r.font.size]
    return max(sizes) if sizes else default

issues = []
for idx, slide in enumerate(prs.slides, 1):
    items = []
    for sh in slide.shapes:
        try:
            l, t = inch(sh.left), inch(sh.top)
            w, h = inch(sh.width), inch(sh.height)
        except Exception:
            continue
        if l < -0.02 or t < -0.02 or l + w > W + 0.02 or t + h > H + 0.02:
            issues.append(f"S{idx} OUT-OF-BOUNDS {sh.shape_type} l={l:.2f} t={t:.2f} r={l+w:.2f} b={t+h:.2f}")
        if sh.has_text_frame:
            tf = sh.text_frame
            text = "\n".join(p.text for p in tf.paragraphs)
            if not text.strip():
                continue
            fs = 12
            for p in tf.paragraphs:
                fs = max(fs, para_font(p))
            # measure each line
            total_h = 0.0
            for p in tf.paragraphs:
                if not p.text.strip():
                    total_h += line_h(fs)
                    continue
                vw = sum(char_w(c, fs) for c in p.text)
                lines = max(1, int(vw / max(w, 0.01)) + (1 if vw % max(w, 0.01) > 0.0001 else 0))
                total_h += lines * line_h(fs)
            if total_h > h + 0.06:
                issues.append(f"S{idx} TEXT-OVERFLOW box_h={h:.2f} need_h={total_h:.2f} w={w:.2f} text={text[:24]!r}")
            items.append((l, t, w, h))
        else:
            items.append((l, t, w, h))
    # text-box vs text-box overlap
    boxes = []
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        txt = "".join(p.text for p in sh.text_frame.paragraphs)
        if not txt.strip():
            continue
        try:
            boxes.append((inch(sh.left), inch(sh.top), inch(sh.width), inch(sh.height), txt[:16]))
        except Exception:
            pass
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            ox = min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])
            oy = min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1])
            if ox > 0.05 and oy > 0.05:
                inter = ox * oy
                small = min(a[2]*a[3], b[2]*b[3])
                if small > 0 and inter / small > 0.25:
                    issues.append(f"S{idx} TEXT-OVERLAP {a[4]!r} <-> {b[4]!r}")

print("ISSUES", len(issues))
for x in issues:
    print(" -", x)
