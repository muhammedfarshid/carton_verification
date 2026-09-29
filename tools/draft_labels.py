"""Merge auto-label detections into draft labels and render review overlays.

Usage: python tools/draft_labels.py autolabel.json
Writes data/labels_draft.json and data/review/<img>.jpg (numbered boxes).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def nms(dets, th=0.5):
    dets = sorted(dets, key=lambda d: -d["score"])
    keep = []
    for d in dets:
        if all(iou(d["box"], k["box"]) < th for k in keep):
            keep.append(d)
    return keep


def contains_frac(a, b):
    """Fraction of b's area inside a."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))


def draft(d):
    W, H = d["size"]
    dets = [x for x in d["gd_items"] if x["score"] > 0.25] + [x for x in d["owl_items"] if x["score"] > 0.2]
    for x in dets:
        x["box"] = [max(0, x["box"][0]), max(0, x["box"][1]), min(W, x["box"][2]), min(H, x["box"][3])]
    dets = nms(dets, 0.45)
    areas = np.array([(b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]) for b in dets])
    if len(dets):
        med = np.median(areas)
        dets = [x for x, a in zip(dets, areas) if 0.3 * med < a < 2.5 * med]
    # drop boxes that contain several others (group boxes)
    dets = [x for x in dets if sum(contains_frac(x["box"], y["box"]) > 0.8 for y in dets if y is not x) < 2]
    return dets


def main(path):
    al = json.load(open(path))
    out, rev = {}, ROOT / "data/review"
    rev.mkdir(parents=True, exist_ok=True)
    for k, d in sorted(al.items()):
        items = draft(d)
        items.sort(key=lambda x: (x["box"][1], x["box"][0]))
        out[k] = {"items": [[round(v) for v in x["box"]] for x in items], "carton": d.get("carton_poly")}
        img = cv2.imread(str(ROOT / f"data/raw/{k}.jpg"))
        if d.get("carton_poly"):
            cv2.polylines(img, [np.array(d["carton_poly"], np.int32)], True, (255, 0, 255), 4)
        for i, b in enumerate(out[k]["items"]):
            cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (0, 255, 0), 3)
            cv2.putText(img, str(i), (b[0] + 5, b[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imwrite(str(rev / f"{k}.jpg"), img)
        print(k, len(items))
    json.dump(out, open(ROOT / "data/labels_draft.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
