import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

malignant = ["mel", "bcc", "akiec"]
benign = ["nv", "bkl", "df", "vasc"]

def map_label(dx):
    return 1 if dx in malignant else 0

def load_metadata(path):
    csv_path = os.path.join(path, "HAM10000_metadata.csv")
    df = pd.read_csv(csv_path)
    df["label"] = df["dx"].apply(map_label)

    imageid_path = {
        os.path.splitext(os.path.basename(x))[0]: x
        for x in glob(os.path.join(path, "*", "*.jpg"))
    }

    df["image_path"] = df["image_id"].map(imageid_path)
    return df, imageid_path

def get_binary_transforms(img_size=224):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform

class HAM10000BinaryDataset(Dataset):
    def __init__(self, df, img_dict, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dict = img_dict
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "image_id"]
        label = self.df.loc[idx, "label"]

        img_path = self.img_dict.get(img_id)
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_id}")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label