from __future__ import annotations

import csv
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(os.environ.get("HBC_HFSS_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "outputs" / "images" / "hfss_normalized_coefficients.png"
CSV_PATH = ROOT / "outputs" / "calibrated_library_full.csv"

W, H = 2600, 1650
BG = "white"
INK = (28, 34, 43)
MUTED = (92, 103, 116)
GRID = (220, 226, 235)
SURFACE = (48, 112, 214)
SURFACE_COND = (16, 145, 132)
IMPLANT = (180, 82, 94)
IMPLANT_STRESS = (126, 86, 176)


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


F_TITLE = font(52, True)
F_H = font(32, True)
F_TEXT = font(24)
F_SMALL = font(20)
F_TINY = font(17)


def read_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_color(row: dict[str, str]) -> tuple[int, int, int]:
    if row["type"] == "surface" and row["condition"] == "rest":
        return SURFACE
    if row["type"] == "surface":
        return SURFACE_COND
    if row["condition"] == "stress":
        return IMPLANT_STRESS
    return IMPLANT


def log_pos(value: float, vmin: float, vmax: float, y0: int, y1: int) -> int:
    value = max(min(value, vmax), vmin)
    lv = math.log10(value)
    lmin = math.log10(vmin)
    lmax = math.log10(vmax)
    frac = (lv - lmin) / (lmax - lmin)
    return int(y1 - frac * (y1 - y0))


def label_lines(label: str) -> list[str]:
    return label.replace("surface_", "surf_").replace("implant", "impl").split("_")


def draw_panel(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    rect: tuple[int, int, int, int],
    vmin: float,
    vmax: float,
) -> None:
    x0, y0, x1, y1 = rect
    draw.text((x0, y0 - 48), title, font=F_H, fill=INK)
    plot_top = y0 + 10
    plot_bottom = y1 - 125
    plot_left = x0 + 78
    plot_right = x1 - 22

    ticks = [vmin, 0.001, 0.01, 0.1, 1.0, 10.0, vmax]
    ticks = sorted({t for t in ticks if vmin <= t <= vmax})
    for tick in ticks:
        y = log_pos(tick, vmin, vmax, plot_top, plot_bottom)
        draw.line([(plot_left, y), (plot_right, y)], fill=GRID, width=1)
        tick_label = f"{tick:g}"
        draw.text((x0, y - 12), tick_label, font=F_TINY, fill=MUTED)
    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=INK, width=2)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=INK, width=2)

    count = len(rows)
    slot = (plot_right - plot_left) / count
    bar_w = min(70, int(slot * 0.58))
    for idx, row in enumerate(rows):
        value = float(row[metric])
        xc = int(plot_left + slot * idx + slot / 2)
        yv = log_pos(value, vmin, vmax, plot_top, plot_bottom)
        baseline_y = log_pos(1.0, vmin, vmax, plot_top, plot_bottom)
        y_top = min(yv, baseline_y)
        y_bot = max(yv, baseline_y)
        draw.rectangle([xc - bar_w // 2, y_top, xc + bar_w // 2, y_bot], fill=row_color(row))
        draw.line([(xc - bar_w // 2, yv), (xc + bar_w // 2, yv)], fill=INK, width=2)
        draw.text((xc - 35, y_top - 26), f"{value:.2g}", font=F_TINY, fill=INK)
        lines = label_lines(row["label"])
        for line_idx, text in enumerate(lines):
            bbox = draw.textbbox((0, 0), text, font=F_TINY)
            draw.text((xc - (bbox[2] - bbox[0]) // 2, plot_bottom + 12 + line_idx * 20), text, font=F_TINY, fill=INK)

    y_ref = log_pos(1.0, vmin, vmax, plot_top, plot_bottom)
    draw.line([(plot_left, y_ref), (plot_right, y_ref)], fill=(30, 30, 30), width=3)


def main() -> None:
    rows = read_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    draw.text((88, 58), "HFSS Normalized Calibration Coefficients", font=F_TITLE, fill=INK)
    draw.text(
        (90, 122),
        "7-case reduced-order package: coupling, service proxy, local field proxy, and receive-power cap proxy",
        font=F_TEXT,
        fill=MUTED,
    )

    panels = [
        ("g_norm", "Energy coupling proxy g_norm", (90, 315, 1250, 875), 1e-4, 25.0),
        ("r_norm", "Data service proxy r_norm", (1360, 315, 2520, 875), 1e-4, 25.0),
        ("chi_norm", "Local field proxy chi_norm", (90, 1030, 1250, 1588), 0.05, 12.0),
        ("p_rx_cap_norm", "Receive power cap proxy", (1360, 1030, 2520, 1588), 1e-5, 10.0),
    ]
    for metric, title, rect, vmin, vmax in panels:
        draw_panel(draw, rows, metric, title, rect, vmin, vmax)

    legend = [
        (SURFACE, "surface rest"),
        (SURFACE_COND, "surface loose/sweat"),
        (IMPLANT, "implant rest"),
        (IMPLANT_STRESS, "implant stress"),
    ]
    x = 90
    for color, text in legend:
        draw.rectangle([x, 182, x + 30, 212], fill=color)
        draw.text((x + 42, 182), text, font=F_SMALL, fill=INK)
        x += 300

    draw.text((90, 1605), "Log-scaled bars; baseline surface_rest = 1.0. Proxies are HFSS scheduling anchors, not SAR certification.", font=F_SMALL, fill=MUTED)
    img.save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
