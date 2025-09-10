from torch.utils.data import Dataset
import pandas as pd
import os


class VideoDataset(Dataset):

    def __init__(self, csv, root, transform=None):
        self.dataframe = pd.read_csv(csv)
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        # label 轉成 int，避免 DataLoader collate 出現 dtype 混亂
        label = int(row.label)

        # 僅拼接一次根目錄與相對路徑
        folder = os.path.normpath(os.path.join(self.root, row.path))

        if self.transform is None:
            raise RuntimeError("transform 不可為 None，需在 transform 中讀取與處理影像資料夾")

        video = self.transform(folder)
        return video, label
