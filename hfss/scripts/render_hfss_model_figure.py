from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(os.environ.get("HBC_HFSS_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "outputs" / "images" / "hfss_model_overview.png"


W, H = 2600, 1500
BG = "white"
INK = (30, 35, 42)
MUTED = (92, 101, 116)
GRID = (226, 232, 240)
SKIN = (238, 151, 145)
FAT = (246, 210, 112)
MUSCLE = (196, 92, 98)
AIR = (210, 228, 246)
COPPER = (222, 148, 38)
PORT = (40, 112, 214)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for item in candidates:
        path = Path(item)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F_TITLE = font(54, True)
F_H = font(38, True)
F_TEXT = font(31)
F_SMALL = font(25)
F_LABEL = font(28, True)


def iso(x: float, y: float, z: float, origin=(680, 510), scale=6.0):
    sx = origin[0] + (x - y) * scale * 0.72
    sy = origin[1] + (x + y) * scale * 0.36 - z * scale
    return sx, sy


def poly(draw: ImageDraw.ImageDraw, points, fill, outline=(110, 118, 130), width=2):
    draw.polygon([(int(x), int(y)) for x, y in points], fill=fill)
    draw.line([(int(x), int(y)) for x, y in points + [points[0]]], fill=outline, width=width)


def cuboid(draw, x0, x1, y0, y1, z0, z1, color, alpha=210, outline=(112, 120, 132)):
    c = color + (alpha,)
    # Top, right, front faces.
    top = [iso(x0, y0, z1), iso(x1, y0, z1), iso(x1, y1, z1), iso(x0, y1, z1)]
    right = [iso(x1, y0, z1), iso(x1, y0, z0), iso(x1, y1, z0), iso(x1, y1, z1)]
    front = [iso(x0, y1, z1), iso(x1, y1, z1), iso(x1, y1, z0), iso(x0, y1, z0)]
    poly(draw, right, c, outline)
    poly(draw, front, tuple(max(v - 25, 0) for v in color) + (alpha,), outline)
    poly(draw, top, tuple(min(v + 24, 255) for v in color) + (alpha,), outline)


def electrode(draw, cx, cy, z, sx=10, sy=6, sz=0.45):
    cuboid(draw, cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, z, z + sz, COPPER, 255, (132, 86, 26))


def arrow(draw, p0, p1, color=INK, width=4):
    draw.line([p0, p1], fill=color, width=width)
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head = [
        (x1, y1),
        (x1 - 22 * ux + 9 * px, y1 - 22 * uy + 9 * py),
        (x1 - 22 * ux - 9 * px, y1 - 22 * uy - 9 * py),
    ]
    draw.polygon(head, fill=color)


def label_box(draw, xy, text, fill=(248, 250, 252), stroke=(190, 198, 210)):
    x, y = xy
    pad_x, pad_y = 16, 10
    bbox = draw.textbbox((x, y), text, font=F_SMALL)
    w = bbox[2] - bbox[0] + 2 * pad_x
    h = bbox[3] - bbox[1] + 2 * pad_y
    draw.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=fill, outline=stroke, width=2)
    draw.text((x + pad_x, y + pad_y - 2), text, font=F_SMALL, fill=INK)


def draw_cross_section(draw):
    x0, y0 = 1460, 360
    width = 760
    scale = 14
    skin_h = int(2 * scale)
    fat_h = int(8 * scale)
    muscle_h = int(50 * scale)
    total_h = skin_h + fat_h + muscle_h

    draw.text((x0, 195), "Cross-section used for HFSS cases", font=F_H, fill=INK)
    draw.rectangle([x0, y0, x0 + width, y0 + skin_h], fill=SKIN, outline=INK, width=2)
    draw.rectangle([x0, y0 + skin_h, x0 + width, y0 + skin_h + fat_h], fill=FAT, outline=INK, width=2)
    draw.rectangle([x0, y0 + skin_h + fat_h, x0 + width, y0 + total_h], fill=MUSCLE, outline=INK, width=2)

    draw.text((x0 + width + 22, y0 - 4), "skin 2 mm", font=F_SMALL, fill=INK)
    draw.text((x0 + width + 22, y0 + skin_h + 36), "fat 8 mm", font=F_SMALL, fill=INK)
    draw.text((x0 + width + 22, y0 + skin_h + fat_h + 230), "muscle 50 mm", font=F_SMALL, fill=INK)

    # Electrodes as side-view rectangles.
    def ex(mm):
        return x0 + int(mm / 120 * width)

    hub_x = ex(35)
    rx_x = ex(85)
    surface_y = y0 - 18
    for cx, name in [(hub_x, "Hub electrodes"), (rx_x, "Surface receiver")]:
        draw.rectangle([cx - 48, surface_y, cx + 48, surface_y + 14], fill=COPPER, outline=(132, 86, 26), width=2)
        draw.text((cx - 96, surface_y - 58), name, font=F_SMALL, fill=INK)
    draw.line([hub_x, y0 - 76, rx_x, y0 - 76], fill=PORT, width=4)
    arrow(draw, (hub_x + 10, y0 - 76), (rx_x - 10, y0 - 76), PORT, 4)
    draw.text((hub_x + 135, y0 - 122), "L = 50 mm", font=F_SMALL, fill=PORT)

    for depth, name in [(10, "implant 10 mm"), (30, "implant 30 mm")]:
        yy = y0 + int(depth * scale)
        draw.rectangle([rx_x - 42, yy - 9, rx_x + 42, yy + 9], fill=COPPER, outline=(132, 86, 26), width=2)
        draw.line([rx_x + 64, y0, rx_x + 64, yy], fill=MUTED, width=3)
        arrow(draw, (rx_x + 64, y0 + 4), (rx_x + 64, yy - 4), MUTED, 3)
        draw.text((rx_x + 88, yy - 18), name, font=F_SMALL, fill=INK)

    draw.rectangle([x0, y0, x0 + width, y0 + total_h], outline=(45, 55, 72), width=4)
    draw.text((x0, y0 + total_h + 32), "Tissue block: 120 mm x 100 mm x 60 mm", font=F_TEXT, fill=INK)
    draw.text((x0, y0 + total_h + 78), "Ports: lumped, 50 ohm; fE=13.56 MHz, fD=40 MHz", font=F_TEXT, fill=INK)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img, "RGBA")

    draw.text((90, 60), "HFSS Tissue-Block Calibration Model", font=F_TITLE, fill=INK)
    draw.text((92, 125), "Reduced-order physical anchor for surface and implantable HBC nodes", font=F_TEXT, fill=MUTED)

    # Light background grid.
    for x in range(80, 1260, 55):
        draw.line([(x, 220), (x + 520, 1020)], fill=GRID, width=1)
    for x in range(320, 1380, 55):
        draw.line([(x, 220), (x - 520, 1020)], fill=GRID, width=1)

    # Air boundary as outline.
    for p0, p1 in [
        (iso(-75, -70, -72), iso(75, -70, -72)),
        (iso(75, -70, -72), iso(75, 70, -72)),
        (iso(75, 70, -72), iso(-75, 70, -72)),
        (iso(-75, 70, -72), iso(-75, -70, -72)),
        (iso(-75, -70, 68), iso(75, -70, 68)),
        (iso(75, -70, 68), iso(75, 70, 68)),
        (iso(75, 70, 68), iso(-75, 70, 68)),
        (iso(-75, 70, 68), iso(-75, -70, 68)),
        (iso(-75, -70, -72), iso(-75, -70, 68)),
        (iso(75, -70, -72), iso(75, -70, 68)),
        (iso(75, 70, -72), iso(75, 70, 68)),
        (iso(-75, 70, -72), iso(-75, 70, 68)),
    ]:
        draw.line([p0, p1], fill=AIR + (160,), width=2)
    label_box(draw, (112, 230), "Radiation boundary / air box")

    # Layered block. Coordinates use z positive upward, skin on top at z=0.
    cuboid(draw, -60, 60, -50, 50, -60, -10, MUSCLE, 215)
    cuboid(draw, -60, 60, -50, 50, -10, -2, FAT, 225)
    cuboid(draw, -60, 60, -50, 50, -2, 0, SKIN, 230)

    # Electrodes.
    electrode(draw, 0, 18, 0.2, sx=10, sy=6, sz=0.8)
    electrode(draw, 0, -18, 0.2, sx=10, sy=6, sz=0.8)
    electrode(draw, 50, 4, 0.2, sx=8, sy=4, sz=0.8)
    electrode(draw, 50, -4, 0.2, sx=8, sy=4, sz=0.8)
    electrode(draw, 50, 4, -10, sx=4, sy=2, sz=1.2)
    electrode(draw, 50, -4, -10, sx=4, sy=2, sz=1.2)
    electrode(draw, 50, 4, -30, sx=4, sy=2, sz=1.2)
    electrode(draw, 50, -4, -30, sx=4, sy=2, sz=1.2)

    label_box(draw, (420, 210), "Port 1: surface hub")
    arrow(draw, (535, 282), iso(0, 18, 3), PORT, 4)
    label_box(draw, (730, 260), "Port 2 cases: surface, 10 mm, 30 mm")
    arrow(draw, (865, 335), iso(50, 4, 0.2), PORT, 4)
    arrow(draw, (920, 380), iso(50, 4, -30), PORT, 4)

    label_box(draw, (180, 970), "Skin")
    label_box(draw, (330, 1020), "Fat")
    label_box(draw, (470, 1080), "Muscle")

    draw.text((92, 1240), "HFSS outputs used downstream: S11, S21, S22; normalized g and r coefficients.", font=F_TEXT, fill=INK)
    draw.text((92, 1285), "Scope note: simplified calibration proxy, not patient-specific SAR or thermal validation.", font=F_TEXT, fill=MUTED)

    draw_cross_section(draw)

    img.save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
