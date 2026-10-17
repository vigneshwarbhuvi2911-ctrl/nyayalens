"""Renders a sample notice as an image-only PDF (no text layer), to exercise the OCR path.

    python scripts/make_scanned_sample.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def render(src: Path, out: Path) -> None:
    lines = []
    for para in src.read_text().splitlines():
        while len(para) > 78:
            cut = para.rfind(" ", 0, 78)
            lines.append(para[:cut])
            para = para[cut + 1:]
        lines.append(para)
    try:
        font = ImageFont.truetype("DejaVuSerif.ttf", 30)
    except OSError:
        font = ImageFont.load_default(size=30)
    img = Image.new("L", (1700, 120 + 44 * len(lines)), 255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((90, 70 + 44 * i), line, fill=20, font=font)
    img.rotate(0.4, fillcolor=255, expand=False).convert("RGB").save(out, "PDF", resolution=200)


if __name__ == "__main__":
    render(ROOT / "data/samples/NOTICE-2.txt", ROOT / "data/scanned/NOTICE-2-scanned.pdf")
    print("written data/scanned/NOTICE-2-scanned.pdf")
