import torch
import torchvision
import cv2
import os
import re


class VideoFolderPathToTensor(object):

    def __init__(self, max_len=None):
        """
        max_len 如未使用可保持 None；此類別維持與既有介面相容：
        __call__(path: str) -> Tensor of shape (T, C, H, W)
        """
        self.max_len = max_len

    @staticmethod
    def _natural_key(p: str):
        """
        依檔名中的數字做自然排序：1.jpg, 2.jpg, ..., 10.jpg
        """
        name = os.path.splitext(os.path.basename(p))[0]
        parts = re.split(r'(\d+)', name)
        return [int(s) if s.isdigit() else s for s in parts]

    def __call__(self, path):
        path = os.path.normpath(path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"資料夾不存在：{path}")

        # 取得該資料夾底下的檔案完整路徑（只做一次 join）
        file_paths = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]

        if len(file_paths) == 0:
            raise RuntimeError(f"資料夾為空，沒有可讀取的影像：{path}")

        # 以自然排序確保時間序列正確（1.jpg < 2.jpg < ... < 10.jpg）
        file_paths = sorted(file_paths, key=self._natural_key)

        # 讀第一張做形狀檢查
        first = cv2.imread(file_paths[0])
        if first is None:
            raise FileNotFoundError(f"cv2.imread 失敗：{file_paths[0]}")
        # BGR -> RGB
        first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)

        num_frames_total = len(file_paths)

        # 時間採樣參數：沿用你的設定
        EXTRACT_FREQUENCY = 1
        num_time_steps = 16  # 與你原本一致

        # 轉換流程：Resize -> ToTensor -> Normalize
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize([224, 224]),
            torchvision.transforms.ToTensor(),  # [0,1]
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225]),
        ])

        # 先建 tensor 容器；channels 由第一張推得
        channels = 3  # RGB
        frames = torch.empty(channels, num_time_steps, 224, 224, dtype=torch.float32)

        last_valid_tensor = None
        for t in range(num_time_steps):
            # 防越界：若索引超過實際影格數，固定用最後一張
            src_index = t * EXTRACT_FREQUENCY
            if src_index >= num_frames_total:
                src_index = num_frames_total - 1

            img = cv2.imread(file_paths[src_index])
            if img is None:
                # 讀不到時：若已有上一張成功影像，沿用上一張；否則以第一張替代
                fallback = last_valid_tensor
                if fallback is None:
                    # 使用第一張（先前已讀成功）
                    # 需重新走相同前處理流程
                    pil = torchvision.transforms.functional.to_pil_image(first)
                    pil = torchvision.transforms.functional.resize(pil, [224, 224])
                    fallback = torchvision.transforms.functional.to_tensor(pil)
                    fallback = torchvision.transforms.functional.normalize(
                        fallback, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    )
                frames[:, t, :, :] = fallback
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 走同一套 transform
            tensor = transform(img)  # (C, H, W)
            frames[:, t, :, :] = tensor
            last_valid_tensor = tensor

        # 回傳 (T, C, H, W) 以符合你原本的介面
        return frames.permute(1, 0, 2, 3)
