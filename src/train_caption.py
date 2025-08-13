
# NOTE: Training large models needs a GPU and time; this is a compact template.

import argparse, os, json, math
from typing import List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

class TinyJsonCaptions(Dataset):
    """
    Expects a JSON lines file or a JSON array with:
      [{"file":"samples/img.jpg","caption":"..."}...]
    Root images dir is --images_dir.
    """
    def __init__(self, json_path: str, images_dir: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.items = data
        self.images_dir = images_dir

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(os.path.join(self.images_dir, it["file"])).convert("RGB")
        return img, it["caption"]

def collate(batch, processor, tok, device):
    imgs, caps = zip(*batch)
    pv = processor(images=list(imgs), return_tensors="pt").pixel_values.to(device)
    labels = tok(list(caps), padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    return pv, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="nlpconnect/vit-gpt2-image-captioning")
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out_dir", default="checkpoints/captioner")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TinyJsonCaptions(args.train_json, args.images_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name).to(device)
    proc = ViTImageProcessor.from_pretrained(args.model_name)
    tok = AutoTokenizer.from_pretrained(args.model_name)

    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    collate_fn=lambda b: collate(b, proc, tok, device))

    optim = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dl) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.1*total_steps), total_steps)

    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    step = 0
    for ep in range(args.epochs):
        for pv, labels in dl:
            out = model(pixel_values=pv, labels=labels)
            loss = out.loss
            loss.backward()
            optim.step(); sched.step(); optim.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"step {step} loss {loss.item():.4f}")
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"Saved to {args.out_dir}")

if __name__ == "__main__":
    main()
