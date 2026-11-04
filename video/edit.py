"""Beat-synced vertical edit of the karate master class (reference-style)."""
import subprocess, sys, math, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from grade import grade_bright as grade

U = '/root/.claude/uploads/5fd69116-c7b5-5348-b8b5-901fa53d91ab/'
CLIP = {
    1: 'e7032f2d-______1.MOV', 2: 'eb63a4c0-_______2.MOV', 3: 'd63c1262-_______3.MOV',
    4: '451138dd-_______4.MOV', 5: '9066a341-5.MOV', 6: 'e0ab67bf-_______6.MOV',
    7: '22bdd81b-7.MOV', 8: '41e89180-8.MOV', 9: 'd706b291-9.MOV',
    10: '06ffb0a5-10.MOV', 11: '0f12286e-11.MOV', 12: '34c6add5-12.MOV', 13: '70fda86e-13.MOV',
}
W, H, FPS = 1080, 1920, 30
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'out.mp4')
MUSIC = os.path.join(HERE, 'music.wav')
FONT = os.path.join(HERE, 'fonts')

def fr(t):
    return int(round(t * FPS))

def read(clip, start, n):
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{start:.3f}', '-i', U + CLIP[clip], '-frames:v', str(n),
           '-vf', f'scale={W}:{H}:flags=lanczos,fps={FPS}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    if len(a) < n:  # pad by holding last frame
        a = np.concatenate([a, np.repeat(a[-1:], n - len(a), 0)])
    return a

def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3

def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)

def zoom(img, z, cx=0.5, cy=0.5):
    if abs(z - 1) < 1e-3:
        return img
    M = np.float32([[z, 0, (1 - z) * W * cx], [0, z, (1 - z) * H * cy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def whip(img, amount, direction):
    """amount 0..1 -> horizontal motion blur + shift."""
    if amount <= 0:
        return img
    k = int(8 + 260 * amount)
    out = cv2.blur(img, (k, 1))
    shift = int(direction * 420 * amount)
    return np.roll(out, shift, axis=1)

def grid(img, s):
    """mirror-tiled mosaic with tile scale s (1 = no effect)."""
    if s >= 0.999:
        return img
    tw, th = max(2, int(W * s)), max(2, int(H * s))
    t = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    block = np.vstack([np.hstack([t, t[:, ::-1]]), np.hstack([t[::-1], t[::-1, ::-1]])])
    nx, ny = math.ceil(W / tw / 2) + 2, math.ceil(H / th / 2) + 2
    mosaic = np.tile(block, (ny, nx, 1))
    # centre an un-flipped tile in the frame
    cx0 = (nx // 2) * 2 * tw
    cy0 = (ny // 2) * 2 * th
    x0 = cx0 + tw // 2 - W // 2
    y0 = cy0 + th // 2 - H // 2
    return mosaic[y0:y0 + H, x0:x0 + W]

# ---------------------------------------------------------------- titles
def font(name, size):
    return ImageFont.truetype(os.path.join(FONT, name + '.ttf'), size)

def text_layer(lines):
    """lines: list of (text, font, fill, y, tracking, align, x). Returns RGBA numpy."""
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for text, f, fill, y, track, align, x in lines:
        widths = [d.textlength(ch, font=f) for ch in text]
        total = sum(widths) + track * (len(text) - 1)
        if align == 'center':
            cx = (W - total) / 2
        else:
            cx = x
        for ch, w in zip(text, widths):
            d.text((cx, y), ch, font=f, fill=fill)
            cx += w + track
    # soft drop shadow under the glyphs
    a = im.getchannel('A')
    sh = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    sh.putalpha(a.filter(ImageFilter.GaussianBlur(14)).point(lambda v: min(255, int(v * 1.6))))
    sh2 = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    sh2.putalpha(a.filter(ImageFilter.GaussianBlur(3)).point(lambda v: int(v * 0.55)))
    out = Image.alpha_composite(Image.alpha_composite(sh, sh2), im)
    return np.asarray(out)

def rect_layer(rects):
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for box, fill in rects:
        d.rectangle(box, fill=fill)
    return np.asarray(im)

def overlay(img, layer, alpha=1.0, dy=0):
    if alpha <= 0:
        return img
    if dy:
        layer = np.roll(layer, dy, axis=0)
    a = layer[..., 3:4].astype(np.float32) / 255 * alpha
    return (img * (1 - a) + layer[..., :3] * a).astype(np.uint8)

def shade(img, top, bottom, strength):
    """darken a horizontal band to make titles readable."""
    g = np.ones((H, 1, 1), np.float32)
    y = np.arange(H)[:, None, None]
    band = np.clip(1 - np.abs((y - (top + bottom) / 2) / ((bottom - top) / 2)), 0, 1) ** 0.7
    g = 1 - strength * band
    return (img * g).astype(np.uint8)

RED = (232, 50, 47, 255)
WHITE = (255, 255, 255, 255)
SOFT = (225, 236, 240, 255)

T_INTRO = text_layer([
    ('МАСТЕР-КЛАСС', font('Unbounded-Black', 84), WHITE, 790, 0, 'center', 0),
    ('ПО КАРАТЭ', font('Unbounded-Black', 60), WHITE, 905, 6, 'center', 0),
])
R_INTRO = rect_layer([((W / 2 - 60, 1010, W / 2, 1020), RED), ((W / 2, 1010, W / 2 + 60, 1020), (43, 123, 255, 255))])
T_NAME = text_layer([
    ('СОФЬЯ БЕРУЛЬЦЕВА', font('Unbounded-Black', 62), WHITE, 1270, 0, 'left', 72),
    ('Бронзовый призёр Олимпийских игр', font('Montserrat-Bold', 40), SOFT, 1360, 0, 'left', 72),
    ('Двукратная чемпионка Азиатских игр', font('Montserrat-Bold', 40), SOFT, 1414, 0, 'left', 72),
])
R_NAME = rect_layer([((72, 1240, 152, 1250), RED)])
T_END = text_layer([
    ('МАСТЕР-КЛАСС ПО КАРАТЭ', font('Montserrat-ExtraBold', 34), SOFT, 790, 7, 'center', 0),
    ('NEOTECH', font('Unbounded-Black', 120), WHITE, 850, 2, 'center', 0),
    ('SCHOOL', font('Unbounded-Black', 120), WHITE, 990, 18, 'center', 0),
    ('Софья Берульцева', font('Montserrat-Bold', 40), SOFT, 1160, 1, 'center', 0),
])
R_END = rect_layer([((W / 2 - 60, 1140, W / 2, 1148), RED), ((W / 2, 1140, W / 2 + 60, 1148), (43, 123, 255, 255))])

def title_alpha(i, a, b, fade_in=8, fade_out=6):
    if i < a or i >= b:
        return 0.0, 0
    p = ease((i - a) / fade_in)
    q = ease((b - i) / fade_out)
    al = min(p, q)
    return al, int(40 * (1 - p))

# ---------------------------------------------------------------- segments
# (start_s, end_s, clip, src_start, options)
SEGS = [
    (2.41, 4.13, 5, 1.55, dict(z=(1.0, 1.10))),
    (4.13, 5.43, 9, 1.90, dict(z=(1.08, 1.0))),
    (5.43, 6.27, 1, 0.40, dict(z=(1.05, 1.12))),
    (6.27, 7.13, 10, 22.6, dict(z=(1.12, 1.04))),
    (7.13, 7.57, 12, 8.6, dict(z=(1.0, 1.06))),
    (7.57, 8.01, 7, 3.3, dict(z=(1.04, 1.1))),
    (8.01, 9.00, 3, 12.8, dict(z=(1.15, 1.06))),
    (9.00, 9.71, 8, 0.6, dict(z=(1.0, 1.08))),
    (9.71, 11.01, 11, 2.3, dict(z=(1.1, 1.0))),
    (11.01, 12.31, 13, 9.6, dict(z=(1.0, 1.05))),
    (12.31, 13.50, 10, 4.2, dict(z=(1.0, 1.08), end=True)),
]
TOTAL = fr(13.50)

rng = np.random.default_rng(7)
def finish(img, i):
    return img
    noise = rng.normal(0, 3.2, (H // 2, W // 2, 1)).astype(np.float32)
    noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def intro_frames():
    n = fr(2.41)
    cut = fr(1.53)
    A = read(7, 0.0, cut)
    B = read(6, 0.0, n - cut)
    for i in range(n):
        if i < cut:
            f = grade(zoom(A[i], 1.0 + 0.08 * i / cut, 0.6, 0.4))
        else:
            k = i - cut
            f = grade(zoom(B[k], 1.08 - 0.06 * k / (n - cut)))
        al, dy = title_alpha(i, fr(0.25), n, 9, 1)
        f = shade(f, 680, 1100, 0.45 * al)
        f = overlay(f, T_INTRO, al, dy)
        f = overlay(f, R_INTRO, al)
        yield f

def seg_frames(start, end, clip, src, o):
    n = fr(end) - fr(start)
    frames = read(clip, src, n)
    z0, z1 = o.get('z', (1.0, 1.0))
    for k in range(n):
        p = k / max(n - 1, 1)
        img = zoom(frames[k], z0 + (z1 - z0) * ease_io(p))
        t = (fr(start) + k) / FPS
        if 'grid' in o:
            g0, g1, g2 = o['grid']
            if g0 <= t < g1:
                s = 1 - (1 - 1 / 3) * ease((t - g0) / 0.18)
            elif g1 <= t < g2:
                s = 1 / 3 + (1 - 1 / 3) * ease_io((t - g1) / (g2 - g1))
            else:
                s = 1.0
            img = grid(img, s)
        img = grade(img, dark=o.get('dark', 0.05))
        K = 4
        if o.get('whip_out') and k >= n - K:
            img = whip(img, (k - (n - K) + 1) / K, o['whip_out'])
        if o.get('whip_in') and k < K:
            img = whip(img, (K - k) / K, o['whip_in'])
        if o.get('fade_in'):
            img = (img * ease(k / 10)).astype(np.uint8)
        if o.get('dip_out') and k >= n - 5:
            img = (img * (1 - (k - (n - 5) + 1) / 5)).astype(np.uint8)
        if clip == 5 and start == 2.41:
            al, dy = title_alpha(k, 6, n - 2, 8, 5)
            img = shade(img, 1150, 1560, 0.5 * al)
            img = overlay(img, T_NAME, al, dy)
            img = overlay(img, R_NAME, al)
        if o.get('end'):
            al, dy = title_alpha(k, 3, n + 99, 9, 1)
            img = (img * (1 - 0.35 * al)).astype(np.uint8)
            img = overlay(img, T_END, al, dy)
            img = overlay(img, R_END, al)
            if k >= n - 8:
                img = (img * (1 - (k - (n - 8) + 1) / 8)).astype(np.uint8)
        yield img

def main():
    enc = subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
         '-i', '-', '-i', MUSIC, '-map', '0:v', '-map', '1:a',
         '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
         '-c:a', 'aac', '-b:a', '192k', '-af', f'afade=t=out:st={TOTAL / FPS - 0.25:.2f}:d=0.25',
         '-t', f'{TOTAL / FPS:.3f}', '-movflags', '+faststart', OUT],
        stdin=subprocess.PIPE)
    count = 0
    for f in intro_frames():
        enc.stdin.write(finish(f, count).tobytes()); count += 1
    for s in SEGS:
        for f in seg_frames(*s):
            enc.stdin.write(finish(f, count).tobytes()); count += 1
        print(f'segment {s[0]:.2f}-{s[1]:.2f} done, frames={count}', flush=True)
    enc.stdin.close(); enc.wait()
    print('total frames', count, 'expected', TOTAL)

if __name__ == '__main__':
    main()
