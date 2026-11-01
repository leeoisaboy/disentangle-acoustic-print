"""Render the README banners and mascot logo with Pillow.

The checked-in PNGs are the deliverables. This script is kept so the visual
identity can be reproduced on a machine with the same public mascot asset and
standard Windows fonts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
MASCOT = ASSETS / "mascot.png"

NAVY = (12, 47, 89)
BLUE = (31, 119, 208)
TEAL = (20, 170, 157)
AMBER = (214, 135, 31)
PURPLE = (110, 91, 199)
MUTED = (88, 105, 122)
GRADIENT = (BLUE, TEAL, AMBER)

FONT_TITLE = Path(r"C:\Windows\Fonts\bahnschrift.ttf")
FONT_UI = Path(r"C:\Windows\Fonts\segoeuib.ttf")
FONT_CN = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_CN_LIGHT = Path(r"C:\Windows\Fonts\msyh.ttc")


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def gradient(size: tuple[int, int], colors: tuple[tuple[int, int, int], ...]) -> Image.Image:
    width, height = size
    stops = np.linspace(0.0, 1.0, len(colors))
    positions = np.linspace(0.0, 1.0, width)
    channels = []
    for channel in range(3):
        values = np.interp(positions, stops, [color[channel] for color in colors])
        channels.append(values.astype(np.uint8))
    row = np.stack(channels, axis=1)[None, :, :].repeat(height, axis=0)
    return Image.fromarray(row, mode="RGB")


def gradient_text(
    canvas: Image.Image,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    colors: tuple[tuple[int, int, int], ...] = GRADIENT,
) -> None:
    left, top, right, bottom = text_font.getbbox(text)
    width = right - left + 12
    height = bottom - top + 12
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).text((6 - left, 6 - top), text, font=text_font, fill=255)
    canvas.paste(gradient(mask.size, colors), (xy[0] - 6, xy[1] - 6), mask)


def fit_font(
    path: Path,
    text: str,
    max_width: int,
    start: int,
    minimum: int,
) -> ImageFont.FreeTypeFont:
    size = start
    while size > minimum:
        candidate = font(path, size)
        if candidate.getbbox(text)[2] <= max_width:
            return candidate
        size -= 2
    return font(path, minimum)


def shadow_card(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int, int] = (255, 255, 255, 230),
    shadow: int = 12,
) -> None:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(
        (x0 + 3, y0 + shadow, x1 + 3, y1 + shadow),
        radius=radius,
        fill=(15, 49, 75, 34),
    )
    overlay = overlay.filter(ImageFilter.GaussianBlur(10))
    canvas.alpha_composite(overlay)
    ImageDraw.Draw(canvas).rounded_rectangle(box, radius=radius, fill=fill)


def pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
) -> int:
    left, top, right, bottom = text_font.getbbox(text)
    width = right - left + 34
    height = bottom - top + 18
    draw.rounded_rectangle(
        (xy[0], xy[1], xy[0] + width, xy[1] + height),
        radius=height // 2,
        fill=(255, 255, 255, 235),
        outline=(*color, 180),
        width=2,
    )
    draw.text((xy[0] + 17, xy[1] + 8), text, font=text_font, fill=color)
    return width


def wave(draw: ImageDraw.ImageDraw, origin: tuple[int, int], width: int, color: tuple[int, int, int]) -> None:
    x0, y0 = origin
    points = []
    for index in range(width):
        x = x0 + index
        phase = index / width
        y = y0 - 18 * np.sin(phase * 6.2 * np.pi) * (0.3 + phase)
        points.append((x, y))
    draw.line(points, fill=(*color, 150), width=3, joint="curve")


def mascot_layer(size: int) -> Image.Image:
    image = Image.open(MASCOT).convert("RGBA")
    ratio = size / image.width
    return image.resize((size, int(image.height * ratio)), Image.Resampling.LANCZOS)


def title_banner(language: str) -> None:
    width, height = 1600, 500
    canvas = Image.new("RGBA", (width, height), (246, 252, 252, 255))
    draw = ImageDraw.Draw(canvas)

    for index in range(height):
        blend = index / height
        color = (
            int(248 - 10 * blend),
            int(253 - 4 * blend),
            int(253 - 6 * blend),
            255,
        )
        draw.line((0, index, width, index), fill=color)

    draw.rounded_rectangle((15, 15, width - 15, height - 15), radius=34, outline=(214, 231, 232, 255), width=2)
    draw.ellipse((1175, 65, 1550, 440), fill=(*TEAL, 24))
    draw.ellipse((1228, 110, 1518, 400), outline=(*TEAL, 70), width=3)

    dog = mascot_layer(265)
    canvas.alpha_composite(dog, (1240, 92))

    label_font = font(FONT_UI, 22)
    draw.text((82, 48), "TRAINING-FREE  /  VOICE-DISENTANGLED  /  NUMPY", font=label_font, fill=TEAL)

    title_font = fit_font(FONT_TITLE, "Disentangle-Acoustic-Print", 1040, 88, 64)
    gradient_text(canvas, (80, 88), "Disentangle-Acoustic-Print", title_font)

    if language == "zh":
        subtitle = "跨音色时序解耦与结构化声学指纹"
        description = "把同一段冻结语音表征拆成静态、运动与证据视图"
        chips = ("纯 NumPy", "免训练", "多尺度在线定位")
        footer = "一只小狗，四条视图，一组能解释失败的对照实验"
    else:
        subtitle = "Cross-voice temporal disentanglement"
        description = "Interpretable acoustic views from one frozen hidden-state trajectory"
        chips = ("NumPy only", "No training", "Online localization")
        footer = "One dog, four views, and a baseline built to explain failures"

    subtitle_font = fit_font(FONT_CN if language == "zh" else FONT_UI, subtitle, 1040, 42, 30)
    draw.text((82, 225), subtitle, font=subtitle_font, fill=NAVY)
    desc_font = font(FONT_CN_LIGHT if language == "zh" else FONT_UI, 25)
    draw.text((84, 288), description, font=desc_font, fill=MUTED)

    chip_font = font(FONT_CN if language == "zh" else FONT_UI, 21)
    cursor = 84
    for index, chip in enumerate(chips):
        cursor += pill(draw, (cursor, 342), chip, chip_font, (BLUE, TEAL, AMBER)[index]) + 12

    wave(draw, (84, 437), 900, BLUE)
    draw.line((84, 415, 190, 415), fill=(*TEAL, 255), width=5)
    draw.line((190, 415, 410, 415), fill=(*BLUE, 255), width=5)
    draw.line((410, 415, 690, 415), fill=(*AMBER, 255), width=5)
    footer_font = font(FONT_CN_LIGHT if language == "zh" else FONT_UI, 19)
    draw.text((84, 458), footer, font=footer_font, fill=(129, 145, 157))

    draw.arc((1080, 135, 1200, 255), 210, 330, fill=NAVY, width=8)
    draw.line((1130, 230, 1070, 300), fill=NAVY, width=8)
    draw.ellipse((1115, 188, 1127, 200), fill=TEAL)
    draw.ellipse((1145, 205, 1157, 217), fill=AMBER)

    canvas.convert("RGB").save(ASSETS / f"title_banner.{language}.png", optimize=True)


def logo() -> None:
    size = 512
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_card(canvas, (12, 12, 500, 500), radius=112, fill=(252, 254, 254, 255), shadow=10)
    draw = ImageDraw.Draw(canvas)
    draw.ellipse((38, 132, 260, 354), fill=(*TEAL, 26))
    draw.ellipse((45, 139, 253, 347), outline=(*TEAL, 85), width=3)
    dog = mascot_layer(178)
    canvas.alpha_composite(dog, (52, 150))

    gradient_text(canvas, (258, 132), "DAP", font(FONT_TITLE, 118), GRADIENT)
    draw.text((262, 268), "ACOUSTIC", font=font(FONT_UI, 25), fill=TEAL)
    draw.text((262, 301), "PRINT", font=font(FONT_UI, 25), fill=AMBER)
    draw.text((262, 365), "OPEN RESEARCH", font=font(FONT_UI, 16), fill=MUTED)
    draw.ellipse((262, 401, 275, 414), fill=TEAL)
    draw.ellipse((292, 401, 305, 414), fill=BLUE)
    draw.ellipse((322, 401, 335, 414), fill=AMBER)

    canvas.save(ASSETS / "logo.png", optimize=True)


def pipeline(language: str) -> None:
    width, height = 1600, 900
    canvas = Image.new("RGBA", (width, height), (247, 252, 252, 255))
    draw = ImageDraw.Draw(canvas)
    for index in range(height):
        blend = index / height
        draw.line(
            (0, index, width, index),
            fill=(int(246 - 6 * blend), int(252 - 3 * blend), int(252 - 4 * blend), 255),
        )
    draw.ellipse((1180, 38, 1545, 230), fill=(*TEAL, 20))
    dog = mascot_layer(135)
    canvas.alpha_composite(dog, (1340, 22))

    title = "一条轨迹，四种看法" if language == "zh" else "One trajectory, four questions"
    font_title = font(FONT_CN if language == "zh" else FONT_TITLE, 56)
    draw.text((64, 58), title, font=font_title, fill=NAVY)
    subtitle = (
        "同一条冻结隐状态轨迹，可以同时回答内容、音色、节奏与边界问题"
        if language == "zh"
        else "A frozen hidden-state path can answer content, voice, timing, and boundary questions at once"
    )
    draw.text((68, 135), subtitle, font=font(FONT_CN_LIGHT if language == "zh" else FONT_UI, 23), fill=MUTED)
    draw.line((68, 178, 460, 178), fill=TEAL, width=5)
    draw.line((460, 178, 660, 178), fill=BLUE, width=5)

    if language == "zh":
        cards = (
            ("01", "隐状态轨迹", ("H ∈ R^(T×D)", "冻结模型", "保留时间顺序"), BLUE),
            ("02", "音色擦除", ("说话人均值", "低秩子空间", "正交投影"), PURPLE),
            ("03", "等弧长视图", ("static", "delta / residual", "velocity"), TEAL),
            ("04", "结构化证据", ("CUSUM change point", "双峰 + 间距", "Z-score 校准"), AMBER),
            ("05", "在线定位", ("单位指纹", "多尺度窗口", "时序 IoU NMS"), (66, 141, 190)),
        )
        bullet_font = font(FONT_CN_LIGHT, 20)
        card_title_font = font(FONT_CN, 25)
        footer = "结论可以错，对照必须留下"
    else:
        cards = (
            ("01", "Hidden path", ("H in R^(T x D)", "frozen model", "ordered frames"), BLUE),
            ("02", "Voice removal", ("speaker centroids", "low-rank subspace", "orthogonal projection"), PURPLE),
            ("03", "Arc-length views", ("static", "delta / residual", "velocity"), TEAL),
            ("04", "Evidence", ("CUSUM change point", "separated peaks", "z-score calibration"), AMBER),
            ("05", "Online search", ("unit fingerprints", "multi-scale windows", "temporal IoU NMS"), (66, 141, 190)),
        )
        bullet_font = font(FONT_UI, 18)
        card_title_font = font(FONT_UI, 24)
        footer = "Conclusions may move. Controls should remain."

    x_positions = (62, 368, 674, 980, 1286)
    for (number, heading, bullets, color), x in zip(cards, x_positions):
        shadow_card(canvas, (x, 230, x + 252, 652), radius=22)
        draw.rounded_rectangle((x, 230, x + 252, 242), radius=6, fill=color)
        draw.ellipse((x + 22, 265, x + 76, 319), fill=color)
        number_font = font(FONT_UI, 22)
        number_width = number_font.getbbox(number)[2]
        draw.text((x + 49 - number_width / 2, 280), number, font=number_font, fill=(255, 255, 255))
        draw.text((x + 20, 340), heading, font=card_title_font, fill=NAVY)
        for row, bullet in enumerate(bullets):
            y = 410 + row * 63
            draw.ellipse((x + 24, y + 7, x + 35, y + 18), fill=color)
            fitted_bullet_font = fit_font(
                FONT_CN_LIGHT if language == "zh" else FONT_UI,
                bullet,
                180,
                bullet_font.size,
                14,
            )
            draw.text((x + 48, y), bullet, font=fitted_bullet_font, fill=(64, 80, 96))
        draw.line((x + 24, 585, x + 228, 585), fill=(222, 232, 235), width=2)

    draw.line((312, 441, 368, 441), fill=(162, 181, 189), width=3)
    draw.line((618, 441, 674, 441), fill=(162, 181, 189), width=3)
    draw.line((924, 441, 980, 441), fill=(162, 181, 189), width=3)
    draw.line((1230, 441, 1286, 441), fill=(162, 181, 189), width=3)

    shadow_card(canvas, (62, 696, 1538, 834), radius=22, fill=(255, 255, 255, 205), shadow=8)
    evidence_font = font(FONT_CN if language == "zh" else FONT_UI, 22)
    sentence = (
        "静态均值给内容锚点，弧长给节奏，结构化证据给边界，NMS 给可读的时间事件。"
        if language == "zh"
        else "Centroid anchors content, arc length stabilizes timing, evidence describes boundaries, NMS keeps events readable."
    )
    draw.text((94, 724), sentence, font=evidence_font, fill=NAVY)
    draw.text((94, 774), footer, font=font(FONT_CN_LIGHT if language == "zh" else FONT_UI, 20), fill=MUTED)
    wave(draw, (92, 806), 1160, TEAL)

    canvas.convert("RGB").save(ASSETS / f"pipeline.{language}.png", optimize=True)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    title_banner("zh")
    title_banner("en")
    logo()
    pipeline("zh")
    pipeline("en")
    print("assets written to", ASSETS)
