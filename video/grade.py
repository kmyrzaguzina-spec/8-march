import numpy as np, cv2

_vig = {}
def vignette(h, w, strength=0.35):
    k = (h, w, strength)
    if k not in _vig:
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / 1.414
        _vig[k] = (1 - strength * np.clip(r, 0, 1) ** 2.2)[..., None]
    return _vig[k]

def grade(rgb, look='teal', dark=0.0):
    """rgb uint8 HxWx3 -> graded uint8. Cinematic teal/cold look tuned for pink mats."""
    f = rgb.astype(np.float32) / 255
    # tame magenta/pink: pull red & blue towards green-neutral where pinkish
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[..., 0] * 2  # 0..360
    sat = hsv[..., 1] / 255
    pink = np.clip(1 - np.minimum(np.abs(hue - 345), 360 - np.abs(hue - 345)) / 50, 0, 1) * np.clip(sat * 1.6, 0, 1)
    lum = (f @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    # overall desaturation, extra on pink
    s = 0.62 - 0.5 * pink[..., None]
    f = lum + (f - lum) * s
    # contrast S-curve
    f = np.clip(f, 0, 1)
    f = f - 0.085 * np.sin(2 * np.pi * f)  # S-curve
    # split toning: teal shadows, slightly warm-neutral highlights
    l = (f @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    sh = (1 - l) ** 2
    f = f + sh * np.array([-0.09, 0.05, 0.08], np.float32) + (l ** 2) * np.array([-0.01, 0.02, 0.02], np.float32)
    # exposure / darken for mood
    f = f * (1 - dark)
    f = f * vignette(*f.shape[:2])
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
