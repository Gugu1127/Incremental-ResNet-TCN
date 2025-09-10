import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from ResTCN import ResTCN

class InferDataset(Dataset):
    def __init__(self, csv_file, root_dir, max_frames=280, size=224):
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"找不到 CSV：{csv_file}")
        self.df = pd.read_csv(csv_file)
        if 'path' not in self.df.columns:
            raise ValueError(f"CSV 缺少欄位 'path'：{csv_file}")
        if 'label' not in self.df.columns:
            # 測試集若無標籤欄位，補上佔位 4
            self.df['label'] = 4

        self.root_dir = root_dir
        self.max_frames = max_frames
        self.tx = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        folder = self.df.iloc[idx]["path"]
        label = int(self.df.iloc[idx]["label"])
        folder_path = os.path.join(self.root_dir, folder)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"找不到影像資料夾：{folder_path}")

        imgs = []
        for i in range(1, self.max_frames+1):
            img_path = os.path.join(folder_path, f"{i}.jpg")
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"影格不存在：{img_path}")
            img = Image.open(img_path).convert("RGB")
            imgs.append(self.tx(img))
        seq = torch.stack(imgs, dim=0)  # [T, C, H, W]
        seq = seq.permute(1, 0, 2, 3)   # -> [C, T, H, W]
        return seq, label, folder

def collate_fn(batch):
    seqs, labels, folders = zip(*batch)
    x = torch.stack(seqs, dim=0)  # [B, C, T, H, W]
    labels = torch.tensor(labels, dtype=torch.long)
    return x, labels, folders

def main():
    parser = argparse.ArgumentParser(description="Inference with pretrained ResTCN.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (models/best_model_xxx.pth)")
    parser.add_argument("--csv", type=str, required=True, help="Path to infer.csv")
    parser.add_argument("--root", type=str, required=True, help="Root folder of Infer_images/<run_key>")
    parser.add_argument("--out", type=str, default="predictions.csv", help="Output CSV file")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference (default=1)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default=0 for max compatibility)")
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"找不到權重檔：{args.ckpt}")
    if not os.path.isdir(args.root):
        raise FileNotFoundError(f"找不到推理影像根目錄：{args.root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    model = ResTCN().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"✅ 已載入權重：{args.ckpt}", flush=True)

    dataset = InferDataset(args.csv, args.root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn
    )

    results = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for x, _, folders in loader:
            x = x.to(device, non_blocking=True)
            # ★ 關鍵：從 [B, C, T, H, W] → [B, T, C, H, W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            outputs = model(x)   # [B, num_classes]
            probs = softmax(outputs)
            preds = torch.argmax(probs, dim=1)
            for i in range(len(folders)):
                row = {"path": folders[i], "pred": int(preds[i])}
                for j in range(probs.size(1)):
                    row[f"prob_{j}"] = float(probs[i, j])
                results.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"✅ 推理完成，輸出: {args.out}", flush=True)

if __name__ == "__main__":
    main()
