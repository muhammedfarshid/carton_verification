"""Carton content verification: carton mask -> item boxes -> product id -> consensus count."""
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from carton.geometry import box_center_in, mask_outside

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"


@dataclass
class Result:
    carton: np.ndarray                      # polygon (N,2)
    boxes: np.ndarray                       # (M,4) all in-carton detections
    scores: np.ndarray
    labels: list                            # product per box
    counted: np.ndarray                     # bool per box: counted toward the target product
    product: str                            # dominant (target) product
    per_class: dict = field(default_factory=dict)
    check_count: int = -1                   # independent SAM2 exemplar count (only with verify=True)

    @property
    def review(self):
        """Primary count and SAM2 cross-check disagree -> flag for a human look."""
        return self.check_count >= 0 and abs(self.check_count - self.count) >= 2

    @property
    def count(self):
        return int(self.counted.sum())

    def status(self, expected):
        return "PASS" if self.count == expected else ("SHORT" if self.count < expected else "OVER")


class Embedder:
    """DINOv2 ViT-S/14 global descriptor for product identification."""

    def __init__(self, device="cpu", name="vit_small_patch14_dinov2.lvd142m"):
        import timm
        import torch
        self.torch, self.device = torch, device
        self.model = timm.create_model(name, pretrained=True, num_classes=0, img_size=224).eval().to(device)
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def prep(self, crop):
        h, w = crop.shape[:2]
        s = 224 / max(h, w)
        c = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
        pad = np.full((224, 224, 3), 114, np.uint8)
        y, x = (224 - c.shape[0]) // 2, (224 - c.shape[1]) // 2
        pad[y:y + c.shape[0], x:x + c.shape[1]] = c
        return ((pad[..., ::-1].astype(np.float32) / 255 - self.mean) / self.std).transpose(2, 0, 1)

    def __call__(self, crops, bs=64):
        if not crops:
            return np.zeros((0, self.model.num_features), np.float32)
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(crops), bs):
                x = self.torch.from_numpy(np.stack([self.prep(c) for c in crops[i:i + bs]])).to(self.device)
                out.append(self.torch.nn.functional.normalize(self.model(x), dim=-1).cpu().numpy())
        return np.concatenate(out)


def rot_variants(crop):
    return [crop, cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(crop, cv2.ROTATE_180),
            cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.flip(crop, 1)]


def build_gallery(embedder, data):
    """data: list of dicts(img, items, product). Returns embeddings and labels (all rotations)."""
    crops, labels = [], []
    for d in data:
        for b in np.asarray(d["items"]).astype(int):
            for c in rot_variants(d["img"][b[1]:b[3], b[0]:b[2]]):
                crops.append(c)
                labels.append(d["product"])
    return embedder(crops), np.array(labels)


def crop_boxes(img, boxes):
    H, W = img.shape[:2]
    out = []
    for x1, y1, x2, y2 in boxes.astype(int):
        out.append(img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)])
    return out


def drop_group_boxes(boxes, scores):
    """Remove boxes that swallow two or more other boxes (a detection of a whole row/group)."""
    keep = []
    for i, a in enumerate(boxes):
        inner = 0
        for j, b in enumerate(boxes):
            if i == j:
                continue
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            if ix / ((b[2] - b[0]) * (b[3] - b[1]) + 1e-6) > 0.8:
                inner += 1
        keep.append(inner < 2)
    keep = np.array(keep, bool)
    return boxes[keep], scores[keep]


class CartonVerifier:
    def __init__(self, weights=WEIGHTS, device=None, item_conf=0.35, carton_conf=0.25,
                 item_imgsz=960, carton_imgsz=640, unknown_sim=0.45, foreign_margin=0.04,
                 verify=False, sam_model="facebook/sam2.1-hiera-large", sam_points=32):
        from ultralytics import YOLO
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        weights = Path(weights)
        self.carton_model = YOLO(str(weights / "carton_seg.pt"))
        self.item_model = YOLO(str(weights / "item_det.pt"))
        self.embedder = Embedder(device)
        g = np.load(weights / "gallery.npz")
        self.g_emb, self.g_lab = g["emb"], g["labels"]
        self.classes = sorted(set(self.g_lab.tolist()))
        self.item_conf, self.carton_conf = item_conf, carton_conf
        self.item_imgsz, self.carton_imgsz = item_imgsz, carton_imgsz
        self.unknown_sim, self.foreign_margin = unknown_sim, foreign_margin
        self.sam = None
        if verify:
            from carton.sam_counter import Sam2Counter
            self.sam = Sam2Counter(device, model_id=sam_model, points_per_side=sam_points)

    # --- stage 1: carton opening -------------------------------------------------
    def find_carton(self, img):
        H, W = img.shape[:2]
        r = self.carton_model.predict(img, imgsz=self.carton_imgsz, conf=self.carton_conf,
                                      device=self.device, verbose=False, retina_masks=True)[0]
        if r.masks is None or len(r.masks) == 0:
            return np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
        best, best_s = None, -1
        cx, cy = W / 2, H / 2
        for m, conf in zip(r.masks.data.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            m = cv2.resize((m > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            c = max(cs, key=cv2.contourArea)
            area = cv2.contourArea(c) / (H * W)
            centred = 1.0 if cv2.pointPolygonTest(c, (cx, cy), False) >= 0 else 0.6
            s = conf * area * centred
            if s > best_s:
                best, best_s = c, s
        hull = cv2.convexHull(best).reshape(-1, 2).astype(np.float32)
        # small outward margin so items touching the carton wall are not cut
        c = hull.mean(0)
        return c + (hull - c) * 1.03

    # --- stage 2: items inside the carton ----------------------------------------
    def detect_items(self, img, carton):
        H, W = img.shape[:2]
        masked = mask_outside(img, carton)
        x1, y1 = np.clip(carton.min(0) - 0.03 * np.array([W, H]), 0, None).astype(int)
        x2, y2 = np.minimum(carton.max(0) + 0.03 * np.array([W, H]), [W, H]).astype(int)
        crop = masked[y1:y2, x1:x2]
        r = self.item_model.predict(crop, imgsz=self.item_imgsz, conf=self.item_conf, iou=0.5,
                                    device=self.device, verbose=False, max_det=300)[0]
        boxes = r.boxes.xyxy.cpu().numpy() + [x1, y1, x1, y1]
        scores = r.boxes.conf.cpu().numpy()
        if len(boxes):
            boxes, scores = drop_group_boxes(boxes, scores)
            keep = np.array([box_center_in(carton, b) for b in boxes], bool)
            boxes, scores = boxes[keep], scores[keep]
        return boxes.reshape(-1, 4), scores

    # --- stage 3+4: product id and consensus -------------------------------------
    def classify(self, img, boxes):
        emb = self.embedder(crop_boxes(img, boxes))
        sims = emb @ self.g_emb.T                                  # (M, G)
        per_class = np.stack([np.sort(sims[:, self.g_lab == c], 1)[:, -5:].mean(1) for c in self.classes], 1)
        return emb, per_class                                       # (M, C) mean of top-5 sims

    def consensus(self, boxes, pc, emb):
        top = pc.argmax(1)
        dom = Counter(top.tolist()).most_common(1)[0][0]
        centroid = emb[top == dom].mean(0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        labels, counted = [], []
        for i in range(len(boxes)):
            is_foreign = top[i] != dom and pc[i, top[i]] - pc[i, dom] > self.foreign_margin
            if pc[i].max() < self.unknown_sim and float(emb[i] @ centroid) < self.unknown_sim:
                labels.append("unknown_item"); counted.append(False)   # nothing we know, nor the carton's product
            elif is_foreign:
                labels.append(self.classes[top[i]]); counted.append(False)
            else:
                labels.append(self.classes[dom]); counted.append(True)
        return dom, labels, np.array(counted, bool)

    def sam2_check(self, img, carton, boxes, scores, counted):
        """Independent count: SAM2 segments inside the carton that match the 3 most confident
        items of the carton's product (DINOv2 similarity + size)."""
        idx = np.where(counted)[0]
        ex = boxes[idx[np.argsort(-scores[idx])[:3]]]
        props = self.sam.proposals(img, carton)
        if len(props):
            props = props[np.array([box_center_in(carton, b) for b in props], bool)]
        sboxes, _ = self.sam.count(img, props, ex, self.embedder)
        return len(sboxes)

    def __call__(self, img):
        carton = self.find_carton(img)
        boxes, scores = self.detect_items(img, carton)
        if len(boxes) == 0:
            return Result(carton, boxes, scores, [], np.zeros(0, bool), "none", {})
        emb, pc = self.classify(img, boxes)
        dom, labels, counted = self.consensus(boxes, pc, emb)
        check = self.sam2_check(img, carton, boxes, scores, counted) if self.sam is not None and counted.any() else -1
        return Result(carton, boxes, scores, labels, counted, self.classes[dom], dict(Counter(labels)), check)
