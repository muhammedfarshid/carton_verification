# Kaggle training entry point. MODE = "cv" (leave-products-out folds) or "final" (all images).
MODE = "cv"

import glob, json, os, shutil, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3", "albumentations>=2.0"], check=False)

SRC = os.path.dirname(glob.glob("/kaggle/input/**/carton/pipeline.py", recursive=True)[0])
ROOT = os.path.dirname(SRC)
WORK = "/kaggle/working"
shutil.copytree(ROOT, f"{WORK}/proj", dirs_exist_ok=True)
os.chdir(f"{WORK}/proj")
sys.path.insert(0, f"{WORK}/proj")

import cv2
import numpy as np
import torch
from carton.augment import load, build, swap_items, item_crops
from carton.pipeline import CartonVerifier, Embedder, build_gallery

NGPU = torch.cuda.device_count()
print("GPUs:", NGPU, torch.cuda.get_device_name(0) if NGPU else "-")
data = load("data/labels.json", "data/raw")

CFG = {
    "cv":    dict(n_item=70, n_carton=40, det_ep=30, seg_ep=30, det_imgsz=800, det_model="yolo11s.pt"),
    "final": dict(n_item=140, n_carton=70, det_ep=45, seg_ep=45, det_imgsz=960, det_model="yolo11s.pt"),
}[MODE]

TRAIN_SNIPPET = """
import sys
from ultralytics import YOLO
m = YOLO(sys.argv[1])
m.train(data=sys.argv[2], epochs=int(sys.argv[3]), imgsz=int(sys.argv[4]), batch=int(sys.argv[5]), device=sys.argv[6],
        project=sys.argv[7], name='run', exist_ok=True, degrees=0, flipud=0.2, fliplr=0.5, mosaic=1.0,
        close_mosaic=5, hsv_h=0.02, patience=100, workers=2, plots=False, verbose=False, cos_lr=True)
"""


def train_pair(ds_dir, out_dir):
    """Train the item detector and the carton segmenter (in parallel when 2 GPUs)."""
    open("/tmp/tr.py", "w").write(TRAIN_SNIPPET)
    det = [sys.executable, "/tmp/tr.py", CFG["det_model"], f"{ds_dir}/items/data.yaml", str(CFG["det_ep"]),
           str(CFG["det_imgsz"]), "12", "0", f"{out_dir}/det"]
    seg = [sys.executable, "/tmp/tr.py", "yolo11n-seg.pt", f"{ds_dir}/carton/data.yaml", str(CFG["seg_ep"]),
           "640", "16", "1" if NGPU > 1 else "0", f"{out_dir}/seg"]
    t = time.time()
    if NGPU > 1:
        ps = [subprocess.Popen(det, stdout=open(f"{out_dir}_det.log", "w"), stderr=subprocess.STDOUT),
              subprocess.Popen(seg, stdout=open(f"{out_dir}_seg.log", "w"), stderr=subprocess.STDOUT)]
        rc = [p.wait() for p in ps]
    else:
        rc = [subprocess.run(c, stdout=open(f"{out_dir}_{n}.log", "w"), stderr=subprocess.STDOUT).returncode
              for c, n in ((det, "det"), (seg, "seg"))]
    print(f"  trained in {time.time() - t:.0f}s rc={rc}")
    for n in ("det", "seg"):
        if rc[0 if n == "det" else 1]:
            print(open(f"{out_dir}_{n}.log").read()[-3000:])
    w = f"{out_dir}/weights"
    os.makedirs(w, exist_ok=True)
    shutil.copy(f"{out_dir}/det/run/weights/best.pt", f"{w}/item_det.pt")
    shutil.copy(f"{out_dir}/seg/run/weights/best.pt", f"{w}/carton_seg.pt")
    return w


def box_pr(pred, gt, th=0.5):
    from carton.augment import iou_matrix
    if len(pred) == 0 or len(gt) == 0:
        return 0, len(pred), len(gt)
    m = iou_matrix(np.asarray(pred, np.float32), np.asarray(gt, np.float32))
    tp, used = 0, set()
    for i in np.argsort(-m.max(1)):
        j = int(m[i].argmax())
        if m[i, j] >= th and j not in used:
            tp += 1; used.add(j)
    return tp, len(pred) - tp, len(gt) - tp


def evaluate(weights, emb, eval_data, gallery_data, tag):
    np.savez(f"{weights}/gallery.npz", **dict(zip(("emb", "labels"), build_gallery(emb, gallery_data))))
    v = CartonVerifier(weights, device="cuda:0")
    v.embedder = emb
    rows = []
    for d in eval_data:
        n = len(d["items"])
        r = v(d["img"])
        tp, fp, fn = box_pr(r.boxes[r.counted], d["items"])
        row = dict(img=d["name"], product=d["product"], gt=n, count=r.count, raw_dets=len(r.boxes),
                   pred_product=r.product, per_class=r.per_class, tp=tp, fp=fp, fn=fn)
        # foreign-item test: swap 2 items for crops of another product -> expected count n-2
        others = [o for o in gallery_data if o["product"] != d["product"]]
        if others:
            import random
            rng = random.Random(0)
            img2 = d["img"].copy()
            idx = rng.sample(range(n), 2)
            src = item_crops([others[0]])
            for i in idx:
                x1, y1, x2, y2 = d["items"][i].astype(int)
                c = src[rng.randrange(len(src))]
                if (c.shape[1] > c.shape[0]) != (x2 - x1 > y2 - y1):
                    c = cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE)
                img2[y1:y2, x1:x2] = cv2.resize(c, (x2 - x1, y2 - y1))
            r2 = v(img2)
            row.update(foreign_expected=n - 2, foreign_count=r2.count, foreign_per_class=r2.per_class)
        print(tag, json.dumps(row))
        rows.append(row)
    return rows


emb = Embedder("cuda:0")
os.makedirs(f"{WORK}/results", exist_ok=True)

if MODE == "cv":
    FOLDS = [("img_01", "img_03", "img_05"), ("img_02", "img_06"), ("img_04", "img_07"), ("img_08", "img_09", "img_10")]
    allrows = []
    for fi, val in enumerate(FOLDS):
        print(f"=== fold {fi}: hold out {val}")
        ds = build(data, f"/tmp/ds{fi}", n_item=CFG["n_item"], n_carton=CFG["n_carton"], val_names=val, seed=fi)
        w = train_pair(ds, f"/tmp/fold{fi}")
        held = [d for d in data if d["name"] in val]
        trainset = [d for d in data if d["name"] not in val]
        # (a) product NOT registered in gallery (brand-new SKU), (b) registered (few reference crops)
        allrows += [dict(r, fold=fi, gallery="unregistered") for r in evaluate(w, emb, held, trainset, "U")]
        allrows += [dict(r, fold=fi, gallery="registered") for r in evaluate(w, emb, held, data, "R")]
        json.dump(allrows, open(f"{WORK}/results/cv.json", "w"), indent=1)
        shutil.rmtree(f"/tmp/ds{fi}", ignore_errors=True)
else:
    ds = build(data, "/tmp/dsall", n_item=CFG["n_item"], n_carton=CFG["n_carton"], val_names=(), seed=42)
    w = train_pair(ds, "/tmp/final")
    rows = evaluate(w, emb, data, data, "F")
    json.dump(rows, open(f"{WORK}/results/final_train.json", "w"), indent=1)
    shutil.copytree(w, f"{WORK}/weights", dirs_exist_ok=True)
    for f in glob.glob("/tmp/final/*/run/results.csv"):
        shutil.copy(f, f"{WORK}/results/{f.split('/')[-3]}_results.csv")
print("DONE")
