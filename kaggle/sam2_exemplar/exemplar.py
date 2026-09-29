# Training-free exemplar counter: SAM2 "segment everything" inside the carton, keep segments that
# look like the product exemplars (DINOv2 similarity + size), dedupe, count.
import glob, json, os, random, subprocess, sys, time

os.environ["SAM2_BUILD_CUDA"] = "0"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                "git+https://github.com/facebookresearch/sam2.git"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hydra-core", "iopath", "albumentations>=2.0"], check=False)

SRC = os.path.dirname(glob.glob("/kaggle/input/**/carton/pipeline.py", recursive=True)[0])
sys.path.insert(0, os.path.dirname(SRC))
os.chdir(os.path.dirname(SRC))

import cv2
import numpy as np
import torch
from carton.augment import load, iou_matrix
from carton.geometry import mask_outside
from carton.pipeline import Embedder, rot_variants
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

data = load("data/labels.json", "data/raw")
by = {d["name"]: d for d in data}
emb = Embedder("cuda")
amg = SAM2AutomaticMaskGenerator.from_pretrained(
    "facebook/sam2.1-hiera-large", points_per_side=32, points_per_batch=128, pred_iou_thresh=0.7,
    stability_score_thresh=0.85, box_nms_thresh=0.7, min_mask_region_area=200)

MAXS = 1024


def carton_crop(d):
    img = mask_outside(d["img"], d["carton"])
    H, W = img.shape[:2]
    x1, y1 = np.clip(d["carton"].min(0), 0, None).astype(int)
    x2, y2 = np.minimum(d["carton"].max(0), [W, H]).astype(int)
    return img[y1:y2, x1:x2], np.array([x1, y1, x1, y1], np.float32)


# cache SAM2 proposals per image (the expensive part), in original image coords
props, sam_time = {}, {}
for d in data:
    crop, off = carton_crop(d)
    s = MAXS / max(crop.shape[:2])
    small = cv2.resize(crop, None, fx=s, fy=s) if s < 1 else crop
    s = min(s, 1.0)
    torch.cuda.synchronize(); t = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks = amg.generate(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    torch.cuda.synchronize(); sam_time[d["name"]] = time.time() - t
    boxes = np.array([[m["bbox"][0], m["bbox"][1], m["bbox"][0] + m["bbox"][2], m["bbox"][1] + m["bbox"][3]]
                      for m in masks], np.float32).reshape(-1, 4) / s + off
    fill = np.array([m["area"] / max(1, m["bbox"][2] * m["bbox"][3]) for m in masks], np.float32)
    crops = [d["img"][int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in boxes]
    ok = [c.size > 0 for c in crops]
    boxes, fill = boxes[ok], fill[ok]
    e = emb([c for c, o in zip(crops, ok) if o])
    props[d["name"]] = dict(boxes=boxes, fill=fill, emb=e)
    print(d["name"], "proposals", len(boxes), f"sam {sam_time[d['name']]:.2f}s")


def exemplar_set(src, k, seed):
    rng = random.Random(seed)
    idx = rng.sample(range(len(src["items"])), k)
    ex = src["items"][idx]
    crops = [c for b in ex.astype(int) for c in rot_variants(src["img"][b[1]:b[3], b[0]:b[2]])]
    wh = ex[:, 2:] - ex[:, :2]
    return emb(crops), wh


def count(name, ex_emb, ex_wh, tau, size_lo=0.35, size_hi=2.5):
    p = props[name]
    if len(p["boxes"]) == 0:
        return np.zeros((0, 4))
    sim = (p["emb"] @ ex_emb.T).max(1)
    wh = p["boxes"][:, 2:] - p["boxes"][:, :2]
    area = wh.prod(1)
    ex_area = np.median(ex_wh.prod(1))
    size_ok = (area > size_lo * ex_area) & (area < size_hi * ex_area)
    keep = (sim >= tau) & size_ok
    b, s = p["boxes"][keep], sim[keep]
    order = np.argsort(-s)
    b, s = b[order], s[order]
    # greedy NMS + drop parts (box mostly inside an already kept box)
    kept = []
    for i in range(len(b)):
        if kept:
            kb = b[kept]
            if iou_matrix(b[i:i + 1], kb).max() > 0.4:
                continue
            ix = np.clip(np.minimum(b[i, 2:], kb[:, 2:]) - np.maximum(b[i, :2], kb[:, :2]), 0, None).prod(1)
            if (ix / (wh[keep][order][i].prod() + 1e-6)).max() > 0.6:
                continue
        kept.append(i)
    return b[kept]


def pr(pred, gt):
    if len(pred) == 0:
        return 0, 0, len(gt)
    m = iou_matrix(np.asarray(pred, np.float32), gt)
    tp, used = 0, set()
    for i in np.argsort(-m.max(1)):
        j = int(m[i].argmax())
        if m[i, j] >= 0.5 and j not in used:
            tp += 1; used.add(j)
    return tp, len(pred) - tp, len(gt) - tp


CROSS = {"img_01": "img_03", "img_03": "img_01", "img_09": "img_10", "img_10": "img_09"}
rows = []
for tau in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
    for d in data:
        n = d["name"]
        for mode, src in (("same", d), ("cross", by.get(CROSS.get(n)))):
            if src is None:
                continue
            res = []
            for seed in range(3):  # 3 random exemplar draws
                e, wh = exemplar_set(src, 3, seed)
                b = count(n, e, wh, tau)
                res.append((len(b),) + pr(b, d["items"]))
            c = [r[0] for r in res]
            rows.append(dict(tau=tau, img=n, mode=mode, gt=len(d["items"]), counts=c,
                             tp=[r[1] for r in res], fp=[r[2] for r in res], fn=[r[3] for r in res]))
os.makedirs("/kaggle/working/results", exist_ok=True)
json.dump(dict(rows=rows, sam_time=sam_time), open("/kaggle/working/results/exemplar.json", "w"), indent=1)
for tau in sorted({r["tau"] for r in rows}):
    R = [r for r in rows if r["tau"] == tau and r["mode"] == "same"]
    ex = np.mean([np.mean([c == r["gt"] for c in r["counts"]]) for r in R])
    mae = np.mean([np.mean([abs(c - r["gt"]) for c in r["counts"]]) for r in R])
    print(f"tau={tau} same: exact={ex:.2f} mae={mae:.2f} | " +
          " ".join(f"{r['img'][-2:]}:{r['counts']}/{r['gt']}" for r in R))
    R = [r for r in rows if r["tau"] == tau and r["mode"] == "cross"]
    print(f"          cross: " + " ".join(f"{r['img'][-2:]}:{r['counts']}/{r['gt']}" for r in R))
print("sam2 sec/img", np.mean(list(sam_time.values())))
print("DONE")
