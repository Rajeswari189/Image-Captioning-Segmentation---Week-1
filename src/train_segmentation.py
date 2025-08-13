
# WARNING: Real training needs hours and a strong GPU; this is a tiny reference loop.

import argparse, os, math
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
from src.datasets.coco_dataset import CocoInstancesDataset

def get_model(num_classes: int = 91):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    # if you wanted to change head for custom classes, you'd re-init here.
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--ann_file", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out_dir", default="checkpoints/seg")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = T.ToTensor()
    ds = CocoInstancesDataset(args.img_dir, args.ann_file, transform=tf)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))

    model = get_model().to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    step = 0
    for ep in range(args.epochs):
        for images, targets in dl:
            images = [i.to(device) for i in images]
            # targets is a tuple of dicts — move tensors to device
            tt = []
            for t in targets:
                td = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items() if v is not None}
                tt.append(td)
            loss_dict = model(images, tt)
            loss = sum(loss_dict.values())
            loss.backward()
            optim.step(); optim.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"step {step} loss {loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "maskrcnn.pth"))
    print(f"Saved weights to {args.out_dir}/maskrcnn.pth")

if __name__ == "__main__":
    main()
