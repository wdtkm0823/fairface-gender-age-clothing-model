import os

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

BASE_DIR = "."
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "fairface-img-margin50-trainval/train")
VALIDATION_IMG_DIR = os.path.join(BASE_DIR, "fairface-img-margin50-trainval/val")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "fairface_label_train.csv")
VALIDATION_CSV_PATH = os.path.join(BASE_DIR, "fairface_label_val.csv")

df = pd.read_csv(TRAIN_CSV_PATH)


class FairFaceDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        # age列（文字列）をカテゴリに変換するためのマッピング例
        # FairFace の age 列は '0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+' など
        self.age_classes = [
            "0-2",
            "3-9",
            "10-19",
            "20-29",
            "30-39",
            "40-49",
            "50-59",
            "60-69",
            "70+",
        ]
        self.age_to_idx = {a: i for i, a in enumerate(self.age_classes)}

        # gender列（male/female）を二値(0,1)にマッピング
        self.gender_to_idx = {"male": 0, "female": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["file"])

        # 画像読み込み
        image = Image.open(img_path).convert("RGB")

        # age, gender を数値に変換
        age_label = self.age_to_idx[row["age"]]
        gender_label = self.gender_to_idx[row["gender"]]

        # transform (データ拡張やサイズ正規化など)
        if self.transform:
            image = self.transform(image)

        return image, age_label, gender_label


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 画像前処理 (学習時)
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 評価時の前処理
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Dataset インスタンス
train_dataset = FairFaceDataset(train_df, TRAIN_IMG_DIR, transform=train_transform)
val_dataset = FairFaceDataset(val_df, VALIDATION_IMG_DIR, transform=val_transform)
