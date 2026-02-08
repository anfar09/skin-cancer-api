import os
from glob import glob

import cv2
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ------------------------------------------------------
# Label Mapping
# ------------------------------------------------------
MALIGNANT = ["mel", "bcc", "akiec"]
BENIGN = ["nv", "bkl", "df", "vasc"]

def map_label(dx: str) -> int:
    """Map diagnosis to binary label: 1 = malignant, 0 = benign."""
    return 1 if dx in MALIGNANT else 0

# ------------------------------------------------------
# Data Loading
# ------------------------------------------------------
def load_metadata(path: str):
    """Load HAM10000 metadata and map image IDs to file paths."""
    csv_path = os.path.join(path, "HAM10000_metadata.csv")
    df = pd.read_csv(csv_path)
    df["label"] = df["dx"].apply(map_label)

    imageid_path = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob(os.path.join(path, "*", "*.jpg"))
    }

    df["image_path"] = df["image_id"].map(imageid_path)
    return df, imageid_path

# ------------------------------------------------------
# Undersampling (Class Balancing)
# ------------------------------------------------------
def balance_dataframe_by_undersample(df: pd.DataFrame) -> pd.DataFrame:
    """Balance dataset by undersampling the majority class."""
    df_minority = df[df["label"] == 1]
    df_majority = df[df["label"] == 0]

    df_majority_downsampled = df_majority.sample(
        n=len(df_minority), random_state=42
    )

    df_balanced = pd.concat([df_minority, df_majority_downsampled]) \
                    .sample(frac=1.0, random_state=42) \
                    .reset_index(drop=True)

    print("After undersampling:")
    print(df_balanced["label"].value_counts())

    return df_balanced

# ------------------------------------------------------
# Transforms (No Augmentation)
# ------------------------------------------------------
def get_binary_transforms(img_size: int = 224):
    """Basic preprocessing (no augmentation)."""
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return transform, transform

# ------------------------------------------------------
# Dataset
# ------------------------------------------------------
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