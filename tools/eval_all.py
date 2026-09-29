"""Run the pipeline on every labelled sample, write annotated outputs and a summary table.

Expected quantity = hand-labelled count (data/labels.json); the pipeline never sees it except for
PASS/SHORT/OVER. Usage: python tools/eval_all.py [--device cpu]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from carton.draw import annotate  # noqa: E402
from carton.pipeline import CartonVerifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default=None)
ap.add_argument("--verify", action="store_true")
a = ap.parse_args()

labels = json.load(open(ROOT / "data/labels.json"))
v = CartonVerifier(device=a.device, verify=a.verify)
out = ROOT / "outputs"
out.mkdir(exist_ok=True)
rows = []
for k, lab in sorted(labels.items()):
    img = cv2.imread(str(ROOT / f"data/raw/{k}.jpg"))
    v(img) if not rows else None  # warm-up
    t = time.perf_counter()
    r = v(img)
    dt = time.perf_counter() - t
    exp = len(lab["items"])
    cv2.imwrite(str(out / f"{k}_annotated.jpg"), annotate(img, r, exp))
    rows.append(dict(img=k, product=lab["product"], pred_product=r.product, expected=exp, count=r.count,
                     status=r.status(exp), per_class=r.per_class, sam2_check=r.check_count, review=r.review, sec=round(dt, 3)))
    print(json.dumps(rows[-1]))
json.dump(rows, open(out / "summary.json", "w"), indent=1)
ok = sum(r["count"] == r["expected"] for r in rows)
print(f"exact count {ok}/{len(rows)}  product correct {sum(r['product'] == r['pred_product'] for r in rows)}/{len(rows)}"
      f"  mean abs err {sum(abs(r['count'] - r['expected']) for r in rows) / len(rows):.2f}"
      f"  mean sec {sum(r['sec'] for r in rows) / len(rows):.2f}")
