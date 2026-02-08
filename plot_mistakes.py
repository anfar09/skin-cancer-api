import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch.nn as nn
import kagglehub
from sklearn.model_selection import train_test_split

from src.data import load_metadata, get_binary_transforms, HAM10000BinaryDataset

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "outputs/models/resnet18_binary_skin_fast.pth"
IMAGE_SIZE = 128

# แสดงภาพ
N_PER_ROW = 8          # จำนวนภาพต่อแถว
N_CORRECT = 8          # แสดงตัวอย่างที่ทายถูก
N_WRONG = 8            # แสดงตัวอย่างที่ทายผิด

# ==========================
# DEVICE
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================
# LOAD DATA (เหมือน main.py)
# ==========================
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
df, imageid_path = load_metadata(path)

_, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
_, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

_, val_tf = get_binary_transforms(IMAGE_SIZE)
test_dataset = HAM10000BinaryDataset(test_df, imageid_path, val_tf)

# ==========================
# LOAD MODEL
# ==========================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ==========================
# HELPER FOR DENORMALIZE
# ==========================
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
classes = ["Benign", "Malignant"]

def denorm(img_tensor):
    img = img_tensor.numpy().transpose(1, 2, 0)
    img = std * img + mean
    return np.clip(img, 0, 1)

# ==========================
# PROFESSIONAL VISUALIZATION
# ==========================
def show_predictions_pro(model, dataset, device):
    correct_imgs = []
    correct_labels = []
    correct_preds = []
    correct_conf = []

    wrong_imgs = []
    wrong_labels = []
    wrong_preds = []
    wrong_conf = []

    # ======== COLLECT IMAGES ========
    for i in range(len(dataset)):
        image, label = dataset[i]
        img = image.unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(img), dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred].item()

        if pred == label and len(correct_imgs) < N_CORRECT:
            correct_imgs.append(denorm(image))
            correct_labels.append(label)
            correct_preds.append(pred)
            correct_conf.append(conf)

        elif pred != label and len(wrong_imgs) < N_WRONG:
            wrong_imgs.append(denorm(image))
            wrong_labels.append(label)
            wrong_preds.append(pred)
            wrong_conf.append(conf)

        if len(correct_imgs) == N_CORRECT and len(wrong_imgs) == N_WRONG:
            break

    # ======== PLOT ========
    fig = plt.figure(figsize=(14, 6))

    # -------- Row 1: Correct --------
    for i in range(N_CORRECT):
        ax = plt.subplot(2, N_PER_ROW, i + 1)
        ax.imshow(correct_imgs[i])
        ax.axis("off")

        title = f"T:{classes[correct_labels[i]]}\nP:{classes[correct_preds[i]]}"
        ax.set_title(title, fontsize=8, color="green")
        ax.text(0.5, -0.15, f"conf: {correct_conf[i]:.2f}",
                ha="center", va="top", transform=ax.transAxes, fontsize=7)

    # -------- Row 2: Wrong --------
    for i in range(N_WRONG):
        ax = plt.subplot(2, N_PER_ROW, N_PER_ROW + i + 1)
        ax.imshow(wrong_imgs[i])
        ax.axis("off")

        title = f"T:{classes[wrong_labels[i]]}\nP:{classes[wrong_preds[i]]}"
        ax.set_title(title, fontsize=8, color="red")
        ax.text(0.5, -0.15, f"conf: {wrong_conf[i]:.2f}",
                ha="center", va="top", transform=ax.transAxes, fontsize=7)

    # ======== Overall titles ========
    fig.suptitle("Model Prediction Overview", fontsize=14, y=0.97)


    plt.tight_layout()
    plt.savefig("outputs/figures/predictions_mistakes.png", dpi=200, bbox_inches="tight")
    plt.show()

# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    show_predictions_pro(model, test_dataset, device)