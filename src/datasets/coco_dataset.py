# Generated: 2025-08-13
# File: src/datasets/coco_dataset.py

import os
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image
from torch.utils.data import Dataset
import json

try:
    from pycocotools.coco import COCO
except Exception:
    COCO = None  # allow running inference without pycocotools installed


def _ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")


class CocoCaptionsDataset(Dataset):
    """
    Minimal COCO captions dataset.
    Works with your mini-COCO:
      IMG_DIR = "data/coco_mini/images_500"
      ANN_FILE = "data/coco_mini/annotations/captions_val2017.json"
    """
    def __init__(self, img_dir: str, ann_file: str, transform=None):
        _ensure_exists(img_dir)
        _ensure_exists(ann_file)
        if COCO is None:
            raise ImportError("pycocotools not installed. `pip install pycocotools`")

        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        file_name = info["file_name"]
        path = os.path.join(self.img_dir, file_name)
        image = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        # many captions per image; pick first
        caption = anns[0]["caption"] if len(anns) else ""

        if self.transform:
            image = self.transform(image)

        return {"image": image, "caption": caption, "image_id": img_id, "file_name": file_name}


class CocoInstancesDataset(Dataset):
    """
    Minimal COCO instances dataset (for segmentation).
    For mini-COCO:
      IMG_DIR = "data/coco_mini/images_500"
      ANN_FILE = "data/coco_mini/annotations/instances_val2017.json"
    """
    def __init__(self, img_dir: str, ann_file: str, transform=None):
        _ensure_exists(img_dir)
        _ensure_exists(ann_file)
        if COCO is None:
            raise ImportError("pycocotools not installed. `pip install pycocotools`")

        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        import numpy as np
        import torch

        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, info["file_name"])
        image = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        for a in anns:
            if "bbox" not in a or "category_id" not in a:
                continue
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
            areas.append(a.get("area", float(w * h)))
            iscrowd.append(a.get("iscrowd", 0))

            if "segmentation" in a:
                rle = self.coco.annToRLE(a)
                m = self.coco.annToMask(a) if rle is None else self.coco.annToMask(a)
                masks.append(m.astype("uint8"))
            else:
                masks.append(None)

        target: Dict[str, Any] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
        }
        if masks and masks[0] is not None:
            masks_np = np.stack(masks, axis=0)  # (N,H,W)
            target["masks"] = torch.from_numpy(masks_np)
        else:
            target["masks"] = None

        if self.transform:
            image = self.transform(image)

        return image, target
