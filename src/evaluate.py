import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device, save_prefix="outputs/figures"):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print("AUC:", auc)
    print("Confusion Matrix:\n", cm)

    # Plot ROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{save_prefix}/roc_curve.png")
    plt.close()

    # Plot Confusion Matrix
    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0,1], ["Benign", "Malignant"])
    plt.yticks([0,1], ["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.title("Confusion Matrix")
    plt.savefig(f"{save_prefix}/confusion_matrix.png")
    plt.close()