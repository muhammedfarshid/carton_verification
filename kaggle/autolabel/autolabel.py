# Draft labels for carton items + carton interior using open-vocabulary detectors.
# Output is reviewed/corrected by hand before training.
import glob, json, os, subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.45"], check=False)

import numpy as np
import torch
from PIL import Image
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor)

dev = "cuda"
src = glob.glob("/kaggle/input/**/img_*.jpg", recursive=True)
src = sorted(src, key=os.path.basename)
print(len(src), src[:2])

PROMPTS = {
    "img_01": ["white box.", "white product box"],
    "img_02": ["black box.", "black product box"],
    "img_03": ["white box.", "white product box"],
    "img_04": ["green ribbon spool in plastic blister.", "green ribbon in clear plastic package"],
    "img_05": ["ribbon roll.", "dark blue ribbon roll"],
    "img_06": ["white label roll.", "white roll"],
    "img_07": ["cardboard box with white label.", "small cardboard box"],
    "img_08": ["yellow ribbon cartridge.", "yellow ribbon cartridge"],
    "img_09": ["small cardboard box with label.", "small cardboard box"],
    "img_10": ["small cardboard box with label.", "small cardboard box"],
}
CARTON = ["open cardboard box.", "open cardboard carton"]

gd_id = "IDEA-Research/grounding-dino-base"
gd_proc = AutoProcessor.from_pretrained(gd_id)
gd = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id).to(dev).eval()
ow_proc = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
ow = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble").to(dev).eval()
sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-huge")
sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(dev).eval()


@torch.no_grad()
def gdino(img, text, box_th=0.2, text_th=0.2):
    inp = gd_proc(images=img, text=text, return_tensors="pt").to(dev)
    out = gd(**inp)
    r = gd_proc.post_process_grounded_object_detection(
        out, inp.input_ids, threshold=box_th, text_threshold=text_th,
        target_sizes=[img.size[::-1]])[0]
    return [dict(box=b.tolist(), score=float(s)) for b, s in zip(r["boxes"], r["scores"])]


@torch.no_grad()
def owl(img, text, th=0.1):
    inp = ow_proc(text=[[text]], images=img, return_tensors="pt").to(dev)
    out = ow(**inp)
    # OWLv2 pads to square; target size is the padded square
    s = max(img.size)
    r = ow_proc.image_processor.post_process_object_detection(out, threshold=th, target_sizes=torch.tensor([[s, s]]))[0]
    return [dict(box=b.tolist(), score=float(sc)) for b, sc in zip(r["boxes"], r["scores"])]


@torch.no_grad()
def sam_mask(img, box):
    inp = sam_proc(img, input_boxes=[[box]], return_tensors="pt").to(dev)
    out = sam(**inp)
    masks = sam_proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
    i = int(out.iou_scores[0, 0].argmax())
    m = masks[0, i].numpy().astype(np.uint8)
    import cv2
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True)
    return c.reshape(-1, 2).tolist()


res = {}
for p in src:
    k = os.path.basename(p)[:-4]
    img = Image.open(p).convert("RGB")
    d = {"size": img.size}
    d["gd_items"] = gdino(img, PROMPTS[k][0])
    d["owl_items"] = owl(img, PROMPTS[k][1])
    d["gd_carton"] = gdino(img, CARTON[0], 0.25, 0.25)
    d["owl_carton"] = owl(img, CARTON[1], 0.05)
    if d["gd_carton"]:
        best = max(d["gd_carton"], key=lambda x: x["score"])
        d["carton_poly"] = sam_mask(img, best["box"])
    res[k] = d
    print(k, len(d["gd_items"]), len(d["owl_items"]), len(d["gd_carton"]))

os.makedirs("/kaggle/working", exist_ok=True)
json.dump(res, open("/kaggle/working/autolabel.json", "w"))
print("done")
