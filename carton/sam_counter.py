"""Exemplar-guided counting with SAM2: segment everything inside the carton, keep the segments that
look like the exemplar items (DINOv2 similarity + size), remove duplicates / parts, count."""
import cv2
import numpy as np

from carton.augment import iou_matrix
from carton.geometry import mask_outside
from carton.pipeline import crop_boxes, rot_variants


class Sam2Counter:
    def __init__(self, device, model_id="facebook/sam2.1-hiera-large", points_per_side=32, max_side=1024,
                 tau=0.5, size_lo=0.35, size_hi=2.5):
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        self.torch, self.device = torch, device
        self.amg = SAM2AutomaticMaskGenerator.from_pretrained(
            model_id, device=device, points_per_side=points_per_side, points_per_batch=64,
            pred_iou_thresh=0.7, stability_score_thresh=0.85, box_nms_thresh=0.7, min_mask_region_area=200)
        self.max_side, self.tau, self.size_lo, self.size_hi = max_side, tau, size_lo, size_hi

    def proposals(self, img, carton):
        H, W = img.shape[:2]
        x1, y1 = np.clip(carton.min(0), 0, None).astype(int)
        x2, y2 = np.minimum(carton.max(0), [W, H]).astype(int)
        crop = mask_outside(img, carton)[y1:y2, x1:x2]
        s = min(1.0, self.max_side / max(crop.shape[:2]))
        small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode():
            if str(self.device).startswith("cuda"):
                with self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                    masks = self.amg.generate(rgb)
            else:
                masks = self.amg.generate(rgb)
        b = np.array([[m["bbox"][0], m["bbox"][1], m["bbox"][0] + m["bbox"][2], m["bbox"][1] + m["bbox"][3]]
                      for m in masks], np.float32).reshape(-1, 4)
        return b / s + np.array([x1, y1, x1, y1], np.float32)

    def count(self, img, props, exemplars, embedder):
        """props: (P,4) SAM2 boxes; exemplars: (K,4) boxes of known-good items. Returns (N,4), (N,) sims."""
        if len(props) == 0 or len(exemplars) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0)
        ex_emb = embedder([c for c in crop_boxes(img, exemplars) for c in rot_variants(c)])
        crops = crop_boxes(img, props)
        ok = np.array([c.size > 0 for c in crops])
        props = props[ok]
        sim = (embedder([c for c, o in zip(crops, ok) if o]) @ ex_emb.T).max(1)
        wh = props[:, 2:] - props[:, :2]
        area = wh.prod(1)
        ex_area = np.median((exemplars[:, 2:] - exemplars[:, :2]).prod(1))
        keep = (sim >= self.tau) & (area > self.size_lo * ex_area) & (area < self.size_hi * ex_area)
        b, s, a = props[keep], sim[keep], area[keep]
        order = np.argsort(-s)
        b, s, a = b[order], s[order], a[order]
        kept = []
        for i in range(len(b)):
            if kept:
                kb = b[kept]
                if iou_matrix(b[i:i + 1], kb).max() > 0.4:
                    continue  # duplicate
                ix = np.clip(np.minimum(b[i, 2:], kb[:, 2:]) - np.maximum(b[i, :2], kb[:, :2]), 0, None).prod(1)
                if (ix / (a[i] + 1e-6)).max() > 0.6:
                    continue  # part of an item already kept
            kept.append(i)
        return b[kept], s[kept]
