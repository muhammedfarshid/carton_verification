"""Synthetic training data from a handful of labelled carton photos.

Two datasets are produced (YOLO format):
  * carton/  - full images, 1 seg class: the carton opening (polygon).
  * items/   - carton-masked crops, 1 box class: "item" (product agnostic).

Item-level copy-paste changes the count in every sample (remove -> SHORT, add -> OVER),
swaps in crops of other products (so the detector stays product agnostic and the
classifier later sees "foreign" items), and draws synthetic hands/arms over the carton.
"""
import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

from carton.geometry import mask_outside, poly_mask

SKIN = [(141, 85, 36), (198, 134, 66), (224, 172, 105), (241, 194, 125), (255, 219, 172), (92, 58, 40)]
GLOVE = [(40, 40, 40), (30, 60, 160), (20, 120, 200), (200, 200, 200)]


def load(labels_path, img_dir):
    labels = json.load(open(labels_path))
    data = []
    for k, v in sorted(labels.items()):
        img = cv2.imread(str(Path(img_dir) / f"{k}.jpg"))
        data.append(dict(name=k, img=img, product=v["product"],
                         carton=np.array(v["carton"], np.float32),
                         items=np.array(v["items"], np.float32)))
    return data


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-6)


def cardboard_patch(img, items, w, h, rng):
    """A patch of plain cardboard (no items) used to 'remove' an item."""
    H, W = img.shape[:2]
    best = None
    for _ in range(60):
        x, y = rng.randint(0, max(0, W - w)), rng.randint(0, max(0, H - h))
        r = np.array([[x, y, x + w, y + h]], np.float32)
        if len(items) and iou_matrix(r, items).max() > 0:
            continue
        p = img[y:y + h, x:x + w]
        hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(0)
        score = -abs(hsv[0] - 15) - abs(hsv[1] - 90) / 3 - p.std() / 4  # brownish, low texture
        if best is None or score > best[0]:
            best = (score, p.copy())
    if best is None:
        return np.full((h, w, 3), (95, 140, 175), np.uint8)
    return best[1]


def remove_items(img, items, frac, rng):
    n = len(items)
    k = max(1, int(round(n * frac)))
    drop = set(rng.sample(range(n), min(k, n - 1)))
    keep = np.array([b for i, b in enumerate(items) if i not in drop], np.float32).reshape(-1, 4)
    out = img.copy()
    for i in drop:
        x1, y1, x2, y2 = items[i].astype(int)
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            continue
        p = cardboard_patch(img, items, w, h, rng)
        p = cv2.resize(p, (w, h))
        # darker = the empty slot is lower than the top layer
        p = np.clip(p.astype(np.float32) * rng.uniform(0.55, 0.95), 0, 255).astype(np.uint8)
        out[y1:y2, x1:x2] = p
    # re-paste kept items that overlapped the removed ones so they stay intact
    for b in keep:
        x1, y1, x2, y2 = b.astype(int)
        if any(iou_matrix(b[None], items[[i]])[0, 0] > 0 for i in drop):
            out[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    return out, keep


def paste_items(img, items, carton, crops, n, rng):
    """Paste item crops into free space inside the carton (OVER scenarios / mixed products)."""
    H, W = img.shape[:2]
    cm = poly_mask(carton, (H, W))
    out, boxes = img.copy(), [b for b in items]
    med = np.median(items[:, 2:] - items[:, :2], 0) if len(items) else np.array([150, 150])
    for _ in range(n):
        crop = rng.choice(crops)
        s = rng.uniform(0.8, 1.2) * np.sqrt(med.prod() / (crop.shape[0] * crop.shape[1]))
        c = cv2.resize(crop, None, fx=s, fy=s)
        h, w = c.shape[:2]
        if h >= H or w >= W:
            continue
        for _ in range(40):
            x, y = rng.randint(0, W - w), rng.randint(0, H - h)
            r = np.array([[x, y, x + w, y + h]], np.float32)
            if cm[y + h // 2, x + w // 2] == 0:
                continue
            if boxes and iou_matrix(r, np.array(boxes)).max() > 0.08:
                continue
            # drop shadow
            sh = out[y + 6:y + h + 6, x + 6:x + w + 6]
            out[y + 6:y + h + 6, x + 6:x + w + 6] = (sh * 0.6).astype(np.uint8)
            out[y:y + h, x:x + w] = c
            boxes.append(r[0])
            break
    return out, np.array(boxes, np.float32).reshape(-1, 4)


def swap_items(img, items, crops, k, rng):
    """Replace k items with crops of (usually different) products."""
    out = img.copy()
    for i in rng.sample(range(len(items)), min(k, len(items))):
        x1, y1, x2, y2 = items[i].astype(int)
        c = rng.choice(crops)
        if (c.shape[1] > c.shape[0]) != (x2 - x1 > y2 - y1):
            c = cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE)
        out[y1:y2, x1:x2] = cv2.resize(c, (x2 - x1, y2 - y1))
    return out


def draw_hand(img, rng):
    """Synthetic arm + hand entering from an image edge. Returns image and occlusion mask."""
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.uint8)
    edge = rng.choice("tblr")
    L = min(H, W)
    t = rng.uniform(0.1, 0.9)
    start = {"t": (t * W, -20), "b": (t * W, H + 20), "l": (-20, t * H), "r": (W + 20, t * H)}[edge]
    reach = rng.uniform(0.25, 0.6) * L
    ang = {"t": 90, "b": -90, "l": 0, "r": 180}[edge] + rng.uniform(-35, 35)
    d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    end = np.array(start) + d * reach
    arm_w = int(rng.uniform(0.07, 0.12) * L)
    cv2.line(m, tuple(np.int32(start)), tuple(np.int32(end)), 255, arm_w)
    cv2.ellipse(m, tuple(np.int32(end + d * arm_w * 0.6)), (int(arm_w * 0.9), int(arm_w * 0.7)),
                ang, 0, 360, 255, -1)
    for f in range(rng.randint(0, 4)):  # fingers
        fa = ang + rng.uniform(-40, 40)
        fd = np.array([np.cos(np.radians(fa)), np.sin(np.radians(fa))])
        p0 = end + d * arm_w * 0.6
        cv2.line(m, tuple(np.int32(p0)), tuple(np.int32(p0 + fd * arm_w * 1.4)), 255, max(3, arm_w // 5))
    col = np.array(rng.choice(SKIN if rng.random() < 0.7 else GLOVE)[::-1], np.float32)
    shade = cv2.GaussianBlur(np.random.default_rng(rng.randint(0, 1 << 30)).normal(1, .08, (H, W)).astype(np.float32), (0, 0), 15)
    layer = np.clip(col[None, None] * shade[..., None], 0, 255)
    soft = cv2.GaussianBlur(m, (0, 0), 3).astype(np.float32)[..., None] / 255
    out = (img * (1 - soft) + layer * soft).astype(np.uint8)
    return out, m


def drop_occluded(items, occ, th=0.6):
    keep = []
    for b in items:
        x1, y1, x2, y2 = b.astype(int)
        r = occ[max(0, y1):y2, max(0, x1):x2]
        if r.size and (r > 0).mean() < th:
            keep.append(b)
    return np.array(keep, np.float32).reshape(-1, 4)


def photometric():
    return A.Compose([
        A.RandomBrightnessContrast(0.35, 0.35, p=0.8),
        A.HueSaturationValue(12, 30, 25, p=0.6),
        A.RandomGamma(p=0.3),
        A.CLAHE(p=0.15),
        A.OneOf([A.MotionBlur(blur_limit=(3, 15)), A.GaussianBlur(blur_limit=(3, 9)), A.Defocus(radius=(2, 5))], p=0.45),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.35),
        A.ISONoise(p=0.2),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3), p=0.3),
        A.ImageCompression(quality_range=(35, 95), p=0.5),
        A.Downscale(scale_range=(0.4, 0.8), p=0.2),
        A.ToGray(p=0.05),
    ])


def geometric(bbox=True):
    kw = dict(bbox_params=A.BboxParams("pascal_voc", label_fields=["cls"], min_visibility=0.45)) if bbox else {}
    return A.Compose([
        A.RandomRotate90(p=0.75),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Affine(scale=(0.7, 1.25), rotate=(-10, 10), shear=(-4, 4), rotate_method="ellipse", translate_percent=(-0.06, 0.06),
                 border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114), p=0.8),
        A.Perspective(scale=(0.02, 0.08), fit_output=True, border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114), p=0.5),
    ], **kw)


def item_crops(data):
    crops = []
    for d in data:
        for b in d["items"].astype(int):
            crops.append(d["img"][b[1]:b[3], b[0]:b[2]].copy())
    return crops


def carton_crop(img, items, carton, rng, jitter=True):
    """Crop to the carton bbox and grey out everything outside the (jittered) carton polygon."""
    H, W = img.shape[:2]
    poly = carton.copy()
    if jitter:  # imperfect carton masks at inference: grow/shrink polygon
        c = poly.mean(0)
        poly = c + (poly - c) * rng.uniform(0.94, 1.08) + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 8, poly.shape)
    if not jitter or rng.random() < 0.85:
        img = mask_outside(img, poly)
    x1, y1 = np.clip(poly.min(0) - 0.03 * np.array([W, H]), 0, None).astype(int)
    x2, y2 = np.minimum(poly.max(0) + 0.03 * np.array([W, H]), [W, H]).astype(int)
    crop = img[y1:y2, x1:x2]
    b = items - np.array([x1, y1, x1, y1], np.float32)
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, x2 - x1); b[:, [1, 3]] = b[:, [1, 3]].clip(0, y2 - y1)
    b = b[((b[:, 2] - b[:, 0]) > 8) & ((b[:, 3] - b[:, 1]) > 8)]
    return crop, b, poly - [x1, y1]


def write_yolo_box(path_img, path_lbl, img, boxes, max_side=1280):
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        boxes = boxes * s
    H, W = img.shape[:2]
    cv2.imwrite(str(path_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(path_lbl, "w") as f:
        for x1, y1, x2, y2 in boxes:
            f.write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")


def make_item_sample(d, crops_all, crops_other, rng):
    img, items = d["img"], d["items"].copy()
    r = rng.random()
    if r < 0.35:
        img, items = remove_items(img, items, rng.uniform(0.05, 0.5), rng)
    elif r < 0.55:
        img, items = paste_items(img, items, d["carton"], crops_all, rng.randint(1, 5), rng)
    if rng.random() < 0.3 and crops_other:
        img = swap_items(img, items, crops_other, rng.randint(1, 3), rng)
    if rng.random() < 0.35:
        img, occ = draw_hand(img, rng)
        items = drop_occluded(items, occ)
    crop, boxes, _ = carton_crop(img, items, d["carton"], rng)
    t = geometric()(image=crop, bboxes=boxes.tolist(), cls=[0] * len(boxes))
    out = photometric()(image=t["image"])["image"]
    return out, np.array(t["bboxes"], np.float32).reshape(-1, 4)


def make_carton_sample(d, rng):
    img = d["img"]
    if rng.random() < 0.3:
        img, _ = draw_hand(img, rng)
    m = poly_mask(d["carton"], img.shape[:2])
    t = geometric(bbox=False)(image=img, mask=m)
    img = photometric()(image=t["image"])["image"]
    return img, t["mask"]


def write_yolo_seg(path_img, path_lbl, img, mask, max_side=1024):
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    H, W = img.shape[:2]
    cv2.imwrite(str(path_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    with open(path_lbl, "w") as f:
        for c in cs:
            if cv2.contourArea(c) < 0.01 * H * W:
                continue
            c = cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True).reshape(-1, 2)
            f.write("0 " + " ".join(f"{x / W:.5f} {y / H:.5f}" for x, y in c) + "\n")


def build(data, out_dir, n_item=100, n_carton=60, val_names=(), seed=0):
    """Generate YOLO datasets. Images in val_names are written un-augmented to val."""
    rng = random.Random(seed)
    out = Path(out_dir)
    train = [d for d in data if d["name"] not in val_names]
    crops_by = {d["name"]: item_crops([d]) for d in train}
    for ds in ("items", "carton"):
        for sp in ("train", "val"):
            (out / ds / "images" / sp).mkdir(parents=True, exist_ok=True)
            (out / ds / "labels" / sp).mkdir(parents=True, exist_ok=True)
    for d in train:
        crops_all = sum(crops_by.values(), [])
        crops_other = [c for k, v in crops_by.items() if k != d["name"] for c in v]
        for i in range(n_item):
            img, boxes = make_item_sample(d, crops_all, crops_other, rng)
            write_yolo_box(out / f"items/images/train/{d['name']}_{i}.jpg",
                           out / f"items/labels/train/{d['name']}_{i}.txt", img, boxes)
        for i in range(n_carton):
            img, m = make_carton_sample(d, rng)
            write_yolo_seg(out / f"carton/images/train/{d['name']}_{i}.jpg",
                           out / f"carton/labels/train/{d['name']}_{i}.txt", img, m)
    val = [d for d in data if d["name"] in val_names] or train
    for d in val:
        crop, boxes, _ = carton_crop(d["img"], d["items"], d["carton"], rng, jitter=False)
        write_yolo_box(out / f"items/images/val/{d['name']}.jpg", out / f"items/labels/val/{d['name']}.txt", crop, boxes)
        write_yolo_seg(out / f"carton/images/val/{d['name']}.jpg", out / f"carton/labels/val/{d['name']}.txt",
                       d["img"], poly_mask(d["carton"], d["img"].shape[:2]))
    for ds in ("items", "carton"):
        (out / ds / "data.yaml").write_text(
            f"path: {(out / ds).resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: {'item' if ds == 'items' else 'carton'}\n")
    return out
