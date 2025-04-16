import os

import pandas as pd

BASE_DIR = "."
IMG_DIR = os.path.join(BASE_DIR, "fairface-img-margin50-trainval/train")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "fairface_label_train.csv")

df = pd.read_csv(TRAIN_CSV_PATH)
