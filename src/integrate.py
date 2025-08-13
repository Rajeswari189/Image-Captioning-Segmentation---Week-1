

import argparse, os, glob
from PIL import Image
from src.models.segmentation import SegmentationModel, overlay_masks
from src.models.captioning import Captioner
from src.utils import apply_mask_to_image, crop_box, label_to_name

def process_image(path: str, out_dir: str, seg: SegmentationModel, cap: Captioner, top_k: int = 6):
    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(path).convert("RGB")
    instances = seg.predict(img)

    # global caption
    global_cap = cap.generate(img)

    # overlay
    overlay = overlay_masks(img, instances[:top_k])
    base = os.path.splitext(os.path.basename(path))[0]
    overlay.save(os.path.join(out_dir, f"{base}_overlay.jpg"))

    # per-instance captions
    lines = [f"GLOBAL: {global_cap}"]
    for i, inst in enumerate(instances[:top_k]):
        name = label_to_name(inst["label"])
        if inst["mask"] is None:
            crop = crop_box(img, inst["box"])
        else:
            masked = apply_mask_to_image(img, inst["mask"])
            crop = crop_box(masked, inst["box"])
        crop_path = os.path.join(out_dir, f"{base}_inst{i}_{name}.jpg")
        crop.save(crop_path)
        lines.append(f"inst{i} [{name}] score={inst['score']:.2f}: {cap.generate(crop)}")

    with open(os.path.join(out_dir, f"{base}_captions.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="folder with images")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--top_k", type=int, default=6)
    args = ap.parse_args()

    seg = SegmentationModel(device=args.device)
    cap = Captioner(device=args.device)

    images = []
    for ext in ("*.jpg","*.jpeg","*.png","*.webp"):
        images.extend(glob.glob(os.path.join(args.images_dir, ext)))
    images.sort()

    if not images:
        raise SystemExit(f"No images found in {args.images_dir}")

    for p in images:
        print(f"Processing {p} ...")
        process_image(p, args.out_dir, seg, cap, args.top_k)

if __name__ == "__main__":
    main()
