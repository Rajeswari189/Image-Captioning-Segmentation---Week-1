# src/utils.py
# Updated: 2025-08-13

from typing import List
import numpy as np
from PIL import Image

COCO_INSTANCE_CATEGORY_NAMES: List[str] = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def label_to_name(label_id: int) -> str:
    if 0 <= label_id < len(COCO_INSTANCE_CATEGORY_NAMES):
        return COCO_INSTANCE_CATEGORY_NAMES[label_id]
    return f"id_{label_id}"

def _ensure_mask_size(mask: np.ndarray, size) -> np.ndarray:
    """Resize binary mask to image.size if needed (nearest)."""
    w, h = size
    mh, mw = mask.shape
    if (mw, mh) == (w, h):
        return mask
    pil_m = Image.fromarray((mask * 255).astype("uint8")).resize((w, h), resample=Image.NEAREST)
    return (np.array(pil_m) > 127).astype("uint8")

def apply_mask_to_image(img_pil: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Keep pixels inside mask, gray out (or mean color) outside.
    Accepts mask in any shape; resizes to image size if needed.
    """
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    arr = np.array(img_pil).astype("uint8")
    mask = _ensure_mask_size(mask, img_pil.size)
    bg = arr.mean(axis=(0, 1)).astype("uint8")
    out = arr.copy()
    out[mask == 0] = bg
    return Image.fromarray(out)

def crop_box(img_pil: Image.Image, box, pad: int = 6) -> Image.Image:
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(img_pil.width,  x2 + pad); y2 = min(img_pil.height, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return img_pil  # fallback
    return img_pil.crop((x1, y1, x2, y2))
