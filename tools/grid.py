"""Render image with coordinate grid (original pixel coords) + labels for manual review."""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
k, lab, out = sys.argv[1], sys.argv[2], sys.argv[3]
img = cv2.imread(str(ROOT / f"data/raw/{k}.jpg"))
H, W = img.shape[:2]
for x in range(0, W, 100):
    cv2.line(img, (x, 0), (x, H), (0, 255, 255) if x % 500 else (0, 0, 255), 1)
    cv2.putText(img, str(x), (x + 2, 18), 0, 0.55, (0, 255, 255), 2)
for y in range(0, H, 100):
    cv2.line(img, (0, y), (W, y), (0, 255, 255) if y % 500 else (0, 0, 255), 1)
    cv2.putText(img, str(y), (2, y - 3), 0, 0.55, (0, 255, 255), 2)
L = json.load(open(lab)).get(k, {})
for i, b in enumerate(L.get("items", [])):
    cv2.rectangle(img, tuple(b[:2]), tuple(b[2:4]), (0, 255, 0), 2)
    cv2.putText(img, str(i), (b[0] + 4, b[1] + 28), 0, 0.9, (0, 0, 255), 2)
if L.get("carton"):
    cv2.polylines(img, [np.array(L["carton"], np.int32)], True, (255, 0, 255), 3)
cv2.imwrite(out, img)
