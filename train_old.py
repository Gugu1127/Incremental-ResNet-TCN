import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
from torch import nn, optim

from ResTCN import ResTCN
from utils import get_dataloader  # 這版只會回傳 {'train': ...}

import argparse

argparse.ArgumentParser(description='Train ResTCN model on preprocessed data.')
# ===== 讀取參數 =====
parser = argparse.ArgumentParser(description='Train ResTCN model on preprocessed data.')
parser.add_argument('--ID', type=int, required=True, help='the ID of the training run')

args = parser.parse_args()
run_id = args.ID
print(f"Running training with ID: {run_id}", flush=True)
# ===== 超參數與環境 =====
torch.manual_seed(0)
num_epochs = 100
batch_size = 16
lr = 0.001
use_cuda = True
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
print("Device being used:", device, flush=True)

# ===== 只建立 train dataloader（不讀 test）=====
dataloader = get_dataloader(batch_size, 'train.csv', os.path.join(os.getcwd(), 'images_train'))
dataset_sizes = {'train': len(dataloader['train'].dataset)}
print(dataset_sizes, flush=True)

# ===== 建立模型 / 優化器 / scheduler / loss =====
model = ResTCN().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
criterion = nn.CrossEntropyLoss().to(device)

# ===== 最佳模型追蹤 =====
best_loss = float('inf')
best_ckpt_path = f'models/best_model_{run_id}.pth'  # 可自行更改路徑

# ===== 訓練迴圈（無測試）=====
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(dataloader['train'], disable=False):
        inputs = inputs.to(device)
        labels = labels.long().view(-1).to(device)  # 確保 shape 為 [B]

        optimizer.zero_grad()
        outputs = model(inputs)                    # 不 squeeze，避免 batch=1 出事
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_sizes['train']
    current_lr = optimizer.param_groups[0]['lr']
    print(f"[train] Epoch: {epoch + 1}/{num_epochs}  Loss: {epoch_loss:.6f}  LR: {current_lr}", flush=True)

    # 若本輪 loss 更低 => 更新最佳並存檔
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'lr': current_lr,
        }, best_ckpt_path)
        print(f"✅ 新最佳 loss: {best_loss:.6f}，已儲存至 {best_ckpt_path}", flush=True)

    # LR 排程（每個 epoch 結束後 step）
    scheduler.step()

print(f"訓練完成。最佳訓練 loss = {best_loss:.6f}，已儲存至 {best_ckpt_path}")
