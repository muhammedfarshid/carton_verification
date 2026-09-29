import cv2
import numpy as np

PALETTE = [(60, 220, 60), (40, 220, 230), (240, 170, 40), (230, 90, 230), (60, 140, 255),
           (255, 255, 90), (160, 110, 255), (120, 255, 190)]
FOREIGN = (40, 40, 235)
STATUS_BG = {"PASS": (60, 160, 40), "SHORT": (50, 50, 210), "OVER": (0, 140, 255)}


def annotate(img, res, expected):
    out = img.copy()
    H, W = out.shape[:2]
    s = max(H, W) / 1600
    th = max(2, int(3 * s))
    cv2.polylines(out, [res.carton.astype(np.int32)], True, (255, 255, 255), th)
    names = [res.product] + sorted(k for k in res.per_class if k != res.product)
    colors = {n: (PALETTE[i % len(PALETTE)] if i == 0 or n != "unknown_item" else FOREIGN) for i, n in enumerate(names)}
    for n in names[1:]:
        colors[n] = FOREIGN
    order = np.lexsort((res.boxes[:, 0], (res.boxes[:, 1] // (0.5 * np.median(res.boxes[:, 3] - res.boxes[:, 1]) + 1)))) \
        if len(res.boxes) else []
    k = 0
    for i in order:
        b = res.boxes[i].astype(int)
        c = colors[res.labels[i]]
        cv2.rectangle(out, tuple(b[:2]), tuple(b[2:]), c, th)
        if res.counted[i]:
            k += 1
            tag = str(k)
        else:
            tag = "X " + res.labels[i]
        fs = 0.8 * s
        (tw, tht), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cv2.rectangle(out, (b[0], b[1]), (b[0] + tw + 8, b[1] + tht + 10), c, -1)
        cv2.putText(out, tag, (b[0] + 4, b[1] + tht + 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th)
    # header
    status = res.status(expected)
    fs = 1.3 * s
    lines = [f"COUNT {res.count} / EXPECTED {expected}  ->  {status}",
             f"product: {res.product}" + (f"   sam2 check: {res.check_count}" if res.check_count >= 0 else "") +
             ("   -> REVIEW" if res.review else "")] + [f"{n}: {res.per_class.get(n, 0)}" + ("  (not counted)" if n != res.product else "")
                                          for n in names]
    lh = int(46 * s)
    hh = int(lh * (len(lines) + 0.6))
    hw = min(W, int(max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, fs if j == 0 else fs * 0.6, th)[0][0]
                        for j, l in enumerate(lines)) + 60 * s))
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (hw, hh), STATUS_BG[status], -1)
    out = cv2.addWeighted(ov, 0.88, out, 0.12, 0)
    y = lh
    for j, l in enumerate(lines):
        f = fs if j == 0 else fs * 0.6
        x = int(12 * s)
        if j >= 2:
            n = names[j - 2]
            cv2.rectangle(out, (x, y - int(22 * s)), (x + int(22 * s), y), colors[n], -1)
            x += int(32 * s)
        cv2.putText(out, l, (x, y), cv2.FONT_HERSHEY_SIMPLEX, f, (255, 255, 255), th if j == 0 else max(1, th - 1))
        y += lh if j == 0 else int(lh * 0.8)
    return out
