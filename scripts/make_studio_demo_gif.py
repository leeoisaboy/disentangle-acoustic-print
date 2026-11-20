"""Render an updated animated demo GIF for Acoustic Print Studio.

Replaces the legacy ``docs/assets/studio-demo.gif`` to showcase:
1. Prominent live microphone entry + floating 60 FPS VU dock
2. Track 01 & 02 seamless 240-frame panoramic horizontal panning (zero seams)
3. Track 03 seamless panoramic 17-tick arc-length ruler + speaker stripping
4. Track 04 seamless 123-D 4-layer fingerprint ribbon + live mic response
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = REPO_ROOT / "docs" / "assets" / "studio-demo.gif"

W, H = 960, 540
FPS = 16

BG = (251, 251, 253)
INK = (29, 29, 31)
INK_SOFT = (66, 66, 69)
MUTED = (110, 110, 115)
FAINT = (134, 134, 139)
BLUE = (0, 113, 227)
BLUE_SOFT = (228, 240, 255)
TEAL = (43, 156, 148)
TEAL_SOFT = (228, 246, 243)
CORAL = (230, 106, 78)
CORAL_SOFT = (253, 236, 231)
VIOLET = (123, 107, 176)
AMBER = (183, 121, 31)
HAIRLINE = (232, 232, 236)

FONT_CN_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_CN = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_MONO = Path(r"C:\Windows\Fonts\consola.ttf")
FONT_MONO_BOLD = Path(r"C:\Windows\Fonts\consolab.ttf")


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


F_EYEBROW = load_font(FONT_MONO_BOLD, 12)
F_HERO = load_font(FONT_CN_BOLD, 38)
F_TITLE = load_font(FONT_CN_BOLD, 28)
F_SUB = load_font(FONT_CN, 13)
F_MONO_SM = load_font(FONT_MONO, 11)
F_MONO_BD = load_font(FONT_MONO_BOLD, 12)
F_CN_SM = load_font(FONT_CN, 12)
F_CN_BD = load_font(FONT_CN_BOLD, 13)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ease_in_out(t: float) -> float:
    x = clamp(t, 0.0, 1.0)
    return -(math.cos(math.pi * x) - 1.0) * 0.5


def speed_color(norm_s: float) -> tuple[int, int, int]:
    s = clamp(norm_s, 0.0, 1.0)
    if s < 0.45:
        u = s / 0.45
        return (int(0 + u * 43), int(113 + u * 43), int(227 - u * 79))
    u = (s - 0.45) / 0.55
    return (int(43 + u * 187), int(156 - u * 50), int(148 - u * 70))


def make_base(mic_live: bool = False) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    # Topbar brand
    draw.rounded_rectangle((36, 16, 58, 38), radius=7, fill=BLUE_SOFT)
    draw.line((40, 27, 44, 27, 46, 21, 49, 33, 52, 24, 54, 27), fill=BLUE, width=2)
    draw.text((66, 19), "Acoustic Print", font=F_CN_BD, fill=INK)
    draw.text((168, 21), "Studio", font=F_MONO_SM, fill=MUTED)

    # Topbar quick-access mic pill
    mx = W - 295
    if mic_live:
        draw.rounded_rectangle((mx, 15, mx + 96, 39), radius=12, fill=CORAL)
        draw.ellipse((mx + 10, 23, mx + 18, 31), fill=(255, 255, 255))
        draw.text((mx + 25, 19), "采集中", font=F_CN_BD, fill=(255, 255, 255))
    else:
        draw.rounded_rectangle((mx, 15, mx + 96, 39), radius=12, fill=CORAL_SOFT)
        draw.ellipse((mx + 10, 23, mx + 18, 31), fill=CORAL)
        draw.text((mx + 25, 19), "实时麦克风", font=F_CN_SM, fill=CORAL)

    # Topbar live telemetry pill
    draw.rounded_rectangle((W - 188, 15, W - 36, 39), radius=12, fill=(242, 242, 246))
    draw.ellipse((W - 176, 23, W - 168, 31), fill=TEAL)
    draw.text((W - 160, 20), "60 FPS / 0.18 ms", font=F_MONO_SM, fill=INK_SOFT)
    return img


def draw_pill(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    text: str,
    bg: tuple[int, int, int],
    fg: tuple[int, int, int],
    font_obj: ImageFont.FreeTypeFont = F_MONO_BD,
) -> None:
    l, t, r, b = draw.textbbox((0, 0), text, font=font_obj)
    tw, th = r - l, b - t
    pw, ph = tw + 20, th + 10
    draw.rounded_rectangle((cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2), radius=ph / 2, fill=bg)
    draw.text((cx - tw / 2, cy - th / 2 - 1), text, font=font_obj, fill=fg)


def draw_mic_dock(draw: ImageDraw.ImageDraw, phase: float, active_text: str = "检测到语音输入 · 正在实时解耦") -> None:
    dx1, dy1, dx2, dy2 = 170, H - 64, W - 170, H - 14
    draw.rounded_rectangle((dx1, dy1, dx2, dy2), radius=18, fill=(255, 255, 255), outline=(244, 214, 206), width=1)
    # Pulse dot
    pr = 5 + 2 * (0.5 + 0.5 * math.sin(phase * 8.0))
    draw.ellipse((dx1 + 22 - pr, dy1 + 25 - pr, dx1 + 22 + pr, dy1 + 25 + pr), fill=CORAL_SOFT)
    draw.ellipse((dx1 + 18, dy1 + 21, dx1 + 26, dy1 + 29), fill=CORAL)
    draw.text((dx1 + 36, dy1 + 10), active_text, font=F_CN_BD, fill=INK)
    draw.text((dx1 + 36, dy1 + 29), "纯浏览器本地 60 FPS 解耦 · 音频不上传", font=F_CN_SM, fill=FAINT)

    # 12-bar live VU equalizer
    vx = dx1 + 290
    for i in range(12):
        lvl = 0.25 + 0.75 * abs(math.sin(phase * 7.5 + i * 0.65) * math.cos(phase * 3.1 - i * 0.3))
        bh = int(6 + lvl * 22)
        by = dy1 + 25 - bh // 2
        col = CORAL if i % 3 == 0 else BLUE
        draw.rounded_rectangle((vx + i * 7, by, vx + i * 7 + 4, by + bh), radius=2, fill=col)

    draw_pill(draw, dx2 - 148, dy1 + 25, "冻结观察", (242, 242, 246), INK_SOFT, F_CN_SM)
    draw_pill(draw, dx2 - 64, dy1 + 25, "录入语音", BLUE, (255, 255, 255), F_CN_SM)


def precompute_panorama() -> tuple[np.ndarray, np.ndarray]:
    """Generate a smooth, continuous 240-frame panoramic waveform + speed profile."""
    n = 240
    t = np.linspace(0.0, 1.0, n)
    # Two natural dialect utterances nestled seamlessly in one continuous 240-frame timeline
    env1 = np.exp(-0.5 * ((t - 0.21) / 0.055) ** 2) + 0.85 * np.exp(-0.5 * ((t - 0.33) / 0.05) ** 2)
    env2 = 0.92 * np.exp(-0.5 * ((t - 0.67) / 0.055) ** 2) + 0.88 * np.exp(-0.5 * ((t - 0.79) / 0.052) ** 2)
    env = env1 + env2 + 0.04
    wave = env * (0.68 * np.sin(2 * math.pi * 19 * t) + 0.32 * np.cos(2 * math.pi * 31 * t))
    wave = np.clip(wave, -0.95, 0.95)
    speed = np.clip(env * 0.82 + 0.12 * np.abs(np.sin(2 * math.pi * 11 * t)), 0.04, 1.0)
    return wave, speed


WAVE_240, SPEED_240 = precompute_panorama()


def render_scene_hero(t: float) -> Image.Image:
    """Scene 1: Hero with prominent Live Microphone CTA + floating live VU dock."""
    img = make_base(mic_live=t > 0.45)
    draw = ImageDraw.Draw(img)

    draw.text((W / 2 - 96, 78), "WHITE-BOX SPEECH ANALYSIS", font=F_EYEBROW, fill=BLUE)
    draw.text((W / 2 - 155, 108), "把解耦过程，", font=F_HERO, fill=INK)
    draw.text((W / 2 - 155, 158), "放回声音本身。", font=F_HERO, fill=FAINT)
    draw.text((W / 2 - 255, 218), "冻结声学轨迹，剥离说话人音色，观察内容如何穿过语速变化。", font=F_SUB, fill=MUTED)

    # Prominent Live Microphone CTA button with breathing ring
    bx1, by1, bx2, by2 = W / 2 - 225, 266, W / 2 + 5, 314
    ring_expand = int(4 + 5 * (0.5 + 0.5 * math.sin(t * 10.0)))
    draw.rounded_rectangle((bx1 - ring_expand, by1 - ring_expand, bx2 + ring_expand, by2 + ring_expand), radius=28, outline=CORAL_SOFT, width=2)
    btn_fill = INK if t > 0.45 else CORAL
    draw.rounded_rectangle((bx1, by1, bx2, by2), radius=24, fill=btn_fill)
    btn_txt = "正在实时分析 · 60 FPS" if t > 0.45 else "开启实时麦克风分析"
    draw.text((bx1 + 38, by1 + 14), btn_txt, font=F_CN_BD, fill=(255, 255, 255))

    # Secondary demo button
    draw.rounded_rectangle((W / 2 + 18, 266, W / 2 + 158, 314), radius=24, fill=BLUE)
    draw.text((W / 2 + 46, 280), "播放演示口令", font=F_CN_BD, fill=(255, 255, 255))

    # LOCAL LIVE callout
    draw.rounded_rectangle((W / 2 - 290, 336, W / 2 + 290, 368), radius=16, fill=(255, 255, 255), outline=(244, 224, 218), width=1)
    draw_pill(draw, W / 2 - 235, 352, "LOCAL LIVE", CORAL_SOFT, CORAL, F_MONO_SM)
    draw.text((W / 2 - 180, 344), "纯浏览器本地 60 FPS 声学特征解耦，开口即见四轨实时响应", font=F_CN_SM, fill=INK_SOFT)

    # Phrase chips
    chips = [("吴语·侬好", True), ("粤语·唔该", False), ("川渝·巴适", False), ("闽南·多谢", False)]
    cx = W / 2 - 210
    for label, active in chips:
        draw_pill(draw, cx + 52, 404, label, BLUE_SOFT if active else (240, 240, 244), BLUE if active else INK_SOFT, F_CN_SM)
        cx += 110

    if t > 0.45:
        draw_mic_dock(draw, t * 2.5)
    return img


def render_scene_pan_wave(t: float) -> Image.Image:
    """Scene 2: Track 01 & 02 Seamless 240-Frame Panoramic Horizontal Pan (Zero Seams!)."""
    img = make_base(mic_live=False)
    draw = ImageDraw.Draw(img)

    draw.text((40, 58), "01 & 02 / SEAMLESS PANORAMA · 240 FRAMES", font=F_EYEBROW, fill=BLUE)
    draw.text((40, 78), "横向全景无缝平移，", font=F_TITLE, fill=INK)
    l, _, r, _ = draw.textbbox((40, 78), "横向全景无缝平移，", font=F_TITLE)
    draw.text((r + 4, 78), "中缝零断层。", font=F_TITLE, fill=FAINT)

    pan = ease_in_out((t - 0.12) / 0.76)  # 0.0 -> 1.0 smoothly crossing 50% midpoint!
    pad_l, pad_r = 44, 44
    view_w = W - pad_l - pad_r
    full_w = view_w * 2.0
    offset_x = pan * view_w

    # Track 01 Waveform (top half)
    mid_y = 212
    draw.line((pad_l, mid_y, W - pad_r, mid_y), fill=HAIRLINE, width=1)

    # Match highlight bands across full 240-frame panorama
    for start_f, end_f, label, col_soft, col_fg in [
        (38, 92, "吴语·侬好 · 0.94", BLUE_SOFT, BLUE),
        (148, 202, "粤语·唔该 · 0.91", TEAL_SOFT, TEAL),
    ]:
        x1 = pad_l + (start_f / 239.0) * full_w - offset_x
        x2 = pad_l + (end_f / 239.0) * full_w - offset_x
        vx1, vx2 = max(pad_l, x1), min(W - pad_r, x2)
        if vx2 > vx1 + 4:
            draw.rounded_rectangle((vx1, 144, vx2, 276), radius=10, fill=col_soft)
            if pad_l + 55 < x1 + 60 < W - pad_r - 55:
                draw_pill(draw, x1 + 66, 156, label, (255, 255, 255), col_fg, F_MONO_SM)

    # Draw continuous 240-frame speed-colored waveform (zero seam at frame 120!)
    pts = []
    for i in range(240):
        x = pad_l + (i / 239.0) * full_w - offset_x
        y = mid_y - WAVE_240[i] * 58
        pts.append((x, y))
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        if x2 < pad_l or x1 > W - pad_r:
            continue
        draw.line((x1, y1, x2, y2), fill=speed_color(SPEED_240[i]), width=2)

    # Track 02 Latent Speed & CUSUM (lower half)
    base_y = 420
    draw.line((pad_l, base_y, W - pad_r, base_y), fill=HAIRLINE, width=1)
    sp_pts = []
    for i in range(240):
        x = pad_l + (i / 239.0) * full_w - offset_x
        y = base_y - SPEED_240[i] * 92
        sp_pts.append((x, y))
    for i in range(len(sp_pts) - 1):
        x1, y1 = sp_pts[i]
        x2, y2 = sp_pts[i + 1]
        if x2 < pad_l or x1 > W - pad_r:
            continue
        draw.line((x1, y1, x2, y2), fill=speed_color(SPEED_240[i]), width=2)

    # Continuous scrubber bar at bottom showing live 0% -> 100% pan
    sy = 468
    draw.text((pad_l, sy - 7), "横向全景浏览", font=F_MONO_SM, fill=FAINT)
    sx1, sx2 = pad_l + 96, W - pad_r - 52
    draw.rounded_rectangle((sx1, sy - 2, sx2, sy + 2), radius=2, fill=HAIRLINE)
    kx = sx1 + int(pan * (sx2 - sx1))
    draw.rounded_rectangle((sx1, sy - 2, kx, sy + 2), radius=2, fill=BLUE)
    draw.ellipse((kx - 7, sy - 7, kx + 7, sy + 7), fill=BLUE)
    draw.text((sx2 + 12, sy - 7), f"{int(pan * 100)}%", font=F_MONO_BD, fill=BLUE)
    draw.text((pad_l, H - 32), "统一 240 帧全景时间轴 · 支持图表直接拖拽平移 · 中缝零断点", font=F_MONO_SM, fill=FAINT)
    return img


def render_scene_ruler(t: float) -> Image.Image:
    """Scene 3: Track 03 Seamless 17-Tick Arc-Length Ruler + Speaker Subspace Stripping."""
    img = make_base(mic_live=False)
    draw = ImageDraw.Draw(img)

    draw.text((40, 58), "03 / TEMPO & TIMBRE", font=F_EYEBROW, fill=BLUE)
    draw.text((40, 78), "换一种语速，", font=F_TITLE, fill=INK)
    l, _, r, _ = draw.textbbox((40, 78), "换一种语速，", font=F_TITLE)
    draw.text((r + 4, 78), "内容仍在原地。", font=F_TITLE, fill=FAINT)

    pan = ease_in_out((t - 0.1) / 0.8)
    pad_l, pad_r = 96, 300
    view_w = W - pad_l - pad_r
    full_w = view_w * 1.9
    offset_x = pan * (full_w - view_w)

    y_lin, y_arc = 195, 365
    draw.line((pad_l, y_lin, W - pad_r, y_lin), fill=(210, 210, 216), width=2)
    draw.line((pad_l, y_arc, W - pad_r, y_arc), fill=TEAL, width=2)

    n_ticks = 17
    for i in range(n_ticks):
        u = i / (n_ticks - 1)
        # Smooth nonlinear arc-length warp
        arc_u = clamp(u + 0.08 * math.sin(2 * math.pi * u * 1.5), 0.0, 1.0)
        lx = pad_l + u * full_w - offset_x
        ax = pad_l + arc_u * full_w - offset_x
        if max(lx, ax) < pad_l - 10 or min(lx, ax) > W - pad_r + 10:
            continue
        # Bezier connector
        pts = []
        mid_y = (y_lin + y_arc) * 0.5
        for s in range(19):
            k = s / 18.0
            mk = 1 - k
            px = mk**3 * lx + 3 * mk**2 * k * lx + 3 * mk * k**2 * ax + k**3 * ax
            py = mk**3 * y_lin + 3 * mk**2 * k * mid_y + 3 * mk * k**2 * mid_y + k**3 * y_arc
            pts.append((px, py))
        shift = abs(ax - lx) / 80.0
        draw.line(pts, fill=speed_color(0.18 + shift * 0.65), width=2)
        if pad_l <= lx <= W - pad_r:
            draw.rectangle((lx - 1.5, y_lin - 7, lx + 1.5, y_lin + 7), fill=FAINT)
        if pad_l <= ax <= W - pad_r:
            draw.ellipse((ax - 7, y_arc - 7, ax + 7, y_arc + 7), fill=TEAL_SOFT)
            draw.ellipse((ax - 4, y_arc - 4, ax + 4, y_arc + 4), fill=TEAL)
            draw.text((ax - 6, y_arc + 12), f"s{i}", font=F_MONO_SM, fill=MUTED)

    # Pinned left labels (never scroll away!)
    draw.rectangle((32, 165, pad_l - 4, 405), fill=BG)
    draw.text((40, y_lin - 7), "Linear", font=F_MONO_BD, fill=FAINT)
    draw.text((40, y_arc - 7), "ArcLen", font=F_MONO_BD, fill=TEAL)

    # Right speaker subspace stripping meter
    rx = W - 268
    draw.text((rx, 240), "说话人子空间投影能量", font=F_CN_SM, fill=MUTED)
    rem = 0.382 + 0.03 * math.sin(t * 6.0)
    bar_w = 220
    w_rem = int(bar_w * rem)
    draw.rounded_rectangle((rx, 268, rx + w_rem - 2, 278), radius=5, fill=CORAL)
    draw.rounded_rectangle((rx + w_rem + 2, 268, rx + bar_w, 278), radius=5, fill=TEAL)
    draw.text((rx, 290), f"滤除音色 {rem * 100:.1f}%", font=F_MONO_BD, fill=CORAL)
    draw.text((rx + 122, 290), f"保留内容 {(1 - rem) * 100:.1f}%", font=F_MONO_BD, fill=TEAL)

    draw.text((40, H - 34), "17 刻度连续等弧长标尺 · 左侧标签固定不随平移滚出", font=F_MONO_SM, fill=FAINT)
    return img


def render_scene_fingerprint_live(t: float) -> Image.Image:
    """Scene 4: Track 04 Seamless 4-Layer Fingerprint Ribbon + Active 60 FPS Live Mic Dock."""
    img = make_base(mic_live=True)
    draw = ImageDraw.Draw(img)

    draw.text((40, 58), "04 / ACOUSTIC FINGERPRINT · LIVE MIC 60 FPS", font=F_EYEBROW, fill=BLUE)
    draw.text((40, 78), "开口说话，", font=F_TITLE, fill=INK)
    l, _, r, _ = draw.textbbox((40, 78), "开口说话，", font=F_TITLE)
    draw.text((r + 4, 78), "四层纹理实时同步。", font=F_TITLE, fill=FAINT)

    pad_l, pad_r = 116, 44
    bar_w = W - pad_l - pad_r
    y1, y2, y3 = 186, 254, 322
    row_h, match_h = 44, 24

    # Segment accent headers
    segs = [
        ("0.45×Static", 0.0, 0.22, BLUE),
        ("0.20×Delta", 0.22, 0.68, TEAL),
        ("0.20×Speed", 0.68, 0.85, AMBER),
        ("0.15×Accel", 0.85, 1.0, VIOLET),
    ]
    for label, s0, s1, col in segs:
        sx0 = pad_l + int(s0 * bar_w)
        sx1 = pad_l + int(s1 * bar_w) - 4
        draw.rounded_rectangle((sx0, 168, sx1, 172), radius=2, fill=col)
        draw.text((sx0 + 2, 148), label, font=F_MONO_SM, fill=col)

    # Smooth continuous ribbons reacting live to voice phase
    steps = 120
    step_w = bar_w / steps
    for s in range(steps):
        u = s / (steps - 1)
        q = math.sin(u * 14.0 + t * 4.5) * math.cos(u * 5.2 - t * 2.2)
        tmpl = math.sin(u * 14.0 + 1.2) * math.cos(u * 5.2 - 0.6)
        match = clamp(1.0 - abs(q - tmpl) * 0.85, 0.0, 1.0)
        x = pad_l + s * step_w

        def tint(val: float) -> tuple[int, int, int]:
            a = 0.12 + 0.78 * abs(val)
            base = BLUE if val >= 0 else CORAL
            return tuple(int(BG[c] * (1 - a) + base[c] * a) for c in range(3))

        draw.rectangle((x, y1, x + step_w + 1, y1 + row_h), fill=tint(q))
        draw.rectangle((x, y2, x + step_w + 1, y2 + row_h), fill=tint(tmpl))
        m_col = TEAL if match > 0.55 else CORAL
        ma = 0.22 + 0.62 * (match if match > 0.55 else (1 - match))
        mc = tuple(int(BG[c] * (1 - ma) + m_col[c] * ma) for c in range(3))
        draw.rectangle((x, y3, x + step_w + 1, y3 + match_h), fill=mc)

    # Pinned row labels
    draw.text((40, y1 + 14), "当前指纹", font=F_CN_BD, fill=INK_SOFT)
    draw.text((40, y2 + 14), "模板指纹", font=F_CN_BD, fill=BLUE)
    draw.text((40, y3 + 4), "逐维共振", font=F_CN_SM, fill=FAINT)

    # Live cosine similarity readout
    sim = 0.86 + 0.08 * math.sin(t * 5.0)
    draw_pill(draw, W - 98, 152, f"余弦 {sim:.3f}", BLUE_SOFT, BLUE)

    # Floating live microphone dock at bottom
    draw_mic_dock(draw, t * 3.0)
    return img


def main() -> None:
    frames: list[Image.Image] = []
    scenes = [
        (render_scene_hero, 32),
        (render_scene_pan_wave, 40),
        (render_scene_ruler, 34),
        (render_scene_fingerprint_live, 34),
    ]
    for fn, count in scenes:
        for i in range(count):
            frames.append(fn(i / max(count - 1, 1)))

    quantized = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        OUT_GIF,
        save_all=True,
        append_images=quantized[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUT_GIF} ({len(quantized)} frames, {OUT_GIF.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
