import os

import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms

BASE_DIR = "."
IMG_DIR = os.path.join(BASE_DIR, "fairface-img-margin50-trainval/train")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "fairface_label_train.csv")

df = pd.read_csv(TRAIN_CSV_PATH)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 画像前処理 (学習時)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 評価時の前処理
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
