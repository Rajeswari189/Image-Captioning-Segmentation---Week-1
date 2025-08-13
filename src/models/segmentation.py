
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw

COCO_INSTANCE_CATEGORY_NAMES = [
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

class SegmentationModel:
    def __init__(self, score_thresh: float = 0.5, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.to(self.device).eval()
        self.score_thresh = score_thresh
        self.transform = T.Compose([T.ToTensor()])

    def predict(self, image: Image.Image):
        """Return list of dicts: {label, score, box, mask}"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        x = self.transform(image).to(self.device)
        with torch.no_grad():
            out = self.model([x])[0]

        results = []
        # Some models may omit masks; handle gracefully
        has_masks = "masks" in out and out["masks"] is not None
        for i in range(len(out["scores"])):
            score = float(out["scores"][i])
            if score < self.score_thresh:
                continue
            box = out["boxes"][i].detach().cpu().numpy().tolist()
            label = int(out["labels"][i])
            mask = None
            if has_masks:
                m = out["masks"][i].detach().cpu().numpy()  # (1,H,W)
                mask = (m[0] > 0.5).astype("uint8")  # (H,W)
            results.append({"label": label, "score": score, "box": box, "mask": mask})
        return results

def overlay_masks(image: Image.Image, instances):
    """Draw boxes + semi-transparent masks."""
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    for inst in instances:
        x1, y1, x2, y2 = map(int, inst["box"])
        label_id = inst["label"]
        label = COCO_INSTANCE_CATEGORY_NAMES[label_id] if 0 <= label_id < len(COCO_INSTANCE_CATEGORY_NAMES) else f"id_{label_id}"
        score = inst["score"]

        # Box + text
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=2)
        draw.text((x1 + 4, max(0, y1 - 14)), f"{label} {score:.2f}", fill=(255, 0, 0, 255))

        # Mask overlay (resize to image size if needed)
        mask = inst.get("mask")
        if mask is not None:
            mh, mw = mask.shape
            if (mw, mh) != image.size:
                from PIL import Image as PILImage
                mask_img = PILImage.fromarray((mask * 255).astype("uint8")).resize(image.size, resample=PILImage.NEAREST)
            else:
                from PIL import Image as PILImage
                mask_img = PILImage.fromarray((mask * 255).astype("uint8"))
            color = Image.new("RGBA", image.size, (255, 0, 0, 80))
            overlay.paste(color, (0, 0), mask=mask_img)

    return overlay.convert("RGB")
