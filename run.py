"""Carton content verification.

    python run.py --image img_02.jpg --expected 12
    -> outputs/img_02_annotated.jpg
"""
import argparse
import json
import time
from pathlib import Path

import cv2

from carton.draw import annotate
from carton.pipeline import CartonVerifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--expected", type=int, required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--device", default=None, help="cpu | cuda (auto if omitted)")
    ap.add_argument("--verify", action="store_true",
                    help="also count with SAM2 (exemplar-guided) and flag REVIEW if the counts disagree; GPU advised")
    ap.add_argument("--sam-model", default="facebook/sam2.1-hiera-large")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"cannot read {a.image}")
    v = CartonVerifier(device=a.device, verify=a.verify, sam_model=a.sam_model)
    v(img)  # warm-up (model load / kernel compile) so the timing below is per-image
    t = time.perf_counter()
    res = v(img)
    dt = time.perf_counter() - t

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(a.image).stem}_annotated.jpg"
    cv2.imwrite(str(out), annotate(img, res, a.expected))
    print(json.dumps({"image": a.image, "product": res.product, "count": res.count, "expected": a.expected,
                      "status": res.status(a.expected), "per_class": res.per_class,
                      "sam2_check": res.check_count if a.verify else None, "review": res.review,
                      "seconds": round(dt, 3), "output": str(out)}, indent=1))


if __name__ == "__main__":
    main()
