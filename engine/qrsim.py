import cv2, numpy as np

LEVELS = {"L": cv2.QRCODE_ENCODER_CORRECT_LEVEL_L, "M": cv2.QRCODE_ENCODER_CORRECT_LEVEL_M,
          "Q": cv2.QRCODE_ENCODER_CORRECT_LEVEL_Q, "H": cv2.QRCODE_ENCODER_CORRECT_LEVEL_H}
TEXT = "https://shipwithpip.com/AT100"


def make(level, version=6, text=TEXT):
    p = cv2.QRCodeEncoder_Params(); p.correction_level = LEVELS[level]; p.version = version
    img = cv2.QRCodeEncoder_create(p).encode(text)
    return img


def modules(level, version=6):
    img = make(level, version)
    n = 17 + 4 * version
    # find module px size: image includes quiet zone of 2 modules? infer
    h = img.shape[0]
    # try quiet zone q so that h = (n + 2q) * m
    for q in (0, 1, 2, 3, 4):
        if h % (n + 2 * q) == 0:
            m = h // (n + 2 * q)
            sub = img[q * m:(q + n) * m:m, q * m:(q + n) * m:m] if q else img[::m, ::m]
            sub = img[q * m + m // 2:(q + n) * m:m, q * m + m // 2:(q + n) * m:m]
            if sub.shape == (n, n):
                return (sub < 128), n
    raise RuntimeError(f"cannot infer grid {img.shape} n={n}")


def render(mat, patch=0, px=12, quiet=4, logo=(255, 90, 170)):
    n = mat.shape[0]
    m = mat.copy()
    canvas = np.full(((n + 2 * quiet) * px,) * 2 + (3,), 255, np.uint8)
    for y in range(n):
        for x in range(n):
            if m[y, x]:
                canvas[(y + quiet) * px:(y + quiet + 1) * px, (x + quiet) * px:(x + quiet + 1) * px] = 0
    if patch > 0:
        a = (n - patch) // 2
        b = a + patch
        canvas[(a + quiet) * px:(b + quiet) * px, (a + quiet) * px:(b + quiet) * px] = logo[::-1]
    return canvas


def scans(canvas, text=TEXT):
    det = cv2.QRCodeDetector()
    v, _, _ = det.detectAndDecode(canvas)
    return v == text


def max_patch(level, version=6):
    mat, n = modules(level, version)
    best = 0
    for k in range(0, n - 14, 1):
        if scans(render(mat, k)):
            best = k
        else:
            # allow check of a few larger sizes (non-monotonic noise)
            if all(not scans(render(mat, kk)) for kk in range(k, min(n - 14, k + 3))):
                break
    return best, n


if __name__ == "__main__":
    for lv in "LMQH":
        mat, n = modules(lv)
        print(lv, "grid", n, "plain scans", scans(render(mat, 0)))
        k, n = max_patch(lv)
        print("   max centered patch", k, "of", n, "->", round(100 * k * k / (n * n), 1), "% of area")
