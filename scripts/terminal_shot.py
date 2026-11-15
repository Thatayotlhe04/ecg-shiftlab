"""Render captured terminal output as a terminal-window PNG.

    python scripts/terminal_shot.py OUT.png "COMMAND" CAPTURED.txt [--cols 66]

The output text is drawn exactly as captured: no lines are added, removed or
recoloured. Only the window frame, the prompt line and a closing empty prompt
are drawn around it.
"""
import argparse

from PIL import Image, ImageDraw, ImageFont

S = 2  # render at 2x for sharp text
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS = "/usr/share/fonts/opentype/inter/Inter-Regular.otf"

BACKDROP = "#0c0c0b"
WINDOW = "#161615"
BORDER = "#2e2e2b"
TITLEBAR = "#1d1d1b"
DOT = "#3d3d3a"
TITLE = "#8a897f"
PROMPT = "#8a897f"
CMD = "#f2f1ee"
OUT = "#d6d5ce"
CURSOR = "#d6d5ce"


def render(out_path, command, text, cols=None, prompt="~/ecg-shiftlab $ ", title="ecg-shiftlab"):
    size = 24 * S
    font = ImageFont.truetype(MONO, size)
    bold = ImageFont.truetype(MONO_B, size)
    tfont = ImageFont.truetype(SANS, 15 * S)
    cw = font.getlength("M")
    lh = int(size * 1.5)

    lines = text.rstrip("\n").split("\n")
    cols = cols or max(max(len(ln) for ln in lines), len(prompt + command)) + 2
    too_long = [ln for ln in lines if len(ln) > cols]
    assert not too_long, f"lines wider than {cols} columns: {too_long}"

    pad_out, pad_in, bar = 56 * S, 34 * S, 46 * S
    win_w = int(cols * cw + 2 * pad_in)
    body_h = (len(lines) + 2) * lh  # prompt line + output + closing prompt
    win_h = bar + pad_in + body_h + pad_in - (lh - size)
    W, H = win_w + 2 * pad_out, win_h + 2 * pad_out

    img = Image.new("RGB", (W, H), BACKDROP)
    d = ImageDraw.Draw(img)
    x0, y0 = pad_out, pad_out
    r = 12 * S
    d.rounded_rectangle([x0, y0, x0 + win_w, y0 + win_h], r, fill=WINDOW, outline=BORDER, width=S)
    d.rounded_rectangle([x0 + S, y0 + S, x0 + win_w - S, y0 + bar], r, fill=TITLEBAR)
    d.rectangle([x0 + S, y0 + bar - r, x0 + win_w - S, y0 + bar], fill=TITLEBAR)
    d.line([x0 + S, y0 + bar, x0 + win_w - S, y0 + bar], fill=BORDER, width=S)
    for i in range(3):
        cx, cy = x0 + 22 * S + i * 20 * S, y0 + bar // 2
        d.ellipse([cx - 6 * S, cy - 6 * S, cx + 6 * S, cy + 6 * S], fill=DOT)
    d.text((x0 + win_w / 2, y0 + bar / 2), title, font=tfont, fill=TITLE, anchor="mm")

    x, y = x0 + pad_in, y0 + bar + pad_in
    d.text((x, y), prompt, font=font, fill=PROMPT)
    d.text((x + len(prompt) * cw, y), command, font=bold, fill=CMD)
    y += lh
    for ln in lines:
        d.text((x, y), ln, font=font, fill=OUT)
        y += lh
    d.text((x, y), prompt, font=font, fill=PROMPT)
    cx = x + len(prompt) * cw
    d.rectangle([cx, y + 3 * S, cx + cw * 0.92, y + size * 1.12], fill=CURSOR)
    img.save(out_path, optimize=True)
    return img.size


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("command")
    ap.add_argument("captured")
    ap.add_argument("--cols", type=int, default=None)
    a = ap.parse_args()
    print(render(a.out, a.command, open(a.captured).read(), cols=a.cols))
