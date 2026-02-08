import os
import kagglehub
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.data import load_metadata, get_binary_transforms, HAM10000BinaryDataset
from src.train import train_one_epoch, validate_one_epoch
from src.evaluate import evaluate_model

def main():

    # ==========================
    # CONFIGURATION
    # ==========================
    BATCH_SIZE = 64
    NUM_WORKERS = 0
    EPOCHS = 150
    LEARNING_RATE = 1e-3
    IMG_SIZE = 128

    # ==========================
    # CREATE OUTPUT DIRECTORIES
    # ==========================
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # ==========================
    # DOWNLOAD DATASET FROM KAGGLE
    # ==========================
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    print("Dataset path:", path)

    df, imageid_path = load_metadata(path)

    # ==========================
    # TRAIN / VAL / TEST SPLIT
    # ==========================
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ==========================
    # TRANSFORMS AND DATASETS
    # ==========================
    train_tf, val_tf = get_binary_transforms(IMG_SIZE)

    train_dataset = HAM10000BinaryDataset(train_df, imageid_path, train_tf)
    val_dataset = HAM10000BinaryDataset(val_df, imageid_path, val_tf)
    test_dataset = HAM10000BinaryDataset(test_df, imageid_path, val_tf)

    # Use only 30% of training data to speed up training
    train_indices = list(range(len(train_dataset)))
    train_indices = train_indices[: int(0.3 * len(train_indices))]
    train_dataset = Subset(train_dataset, train_indices)

    # ==========================
    # DATA LOADERS
    # ==========================
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # ==========================
    # MODEL SETUP
    # ==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = torchvision.models.resnet18(pretrained=True)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # ==========================
    # TRACK METRICS FOR OVERFITTING CHECK
    # ==========================
    train_losses = []
    val_losses = []
    val_accs = []

    # ==========================
    # TRAINING LOOP
    # ==========================
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_acc:.4f}")

    # Save training stats
    torch.save({
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accs": val_accs
    }, "outputs/figures/training_stats.pth")

    print("Saved training stats to outputs/figures/training_stats.pth")

    # ==========================
    # MODEL EVALUATION
    # ==========================
    os.makedirs("outputs/figures", exist_ok=True)

    print("\nEvaluating on test set...")
    evaluate_model(model, test_loader, device, save_prefix="outputs/figures")

    # ==========================
    # SAVE FINAL MODEL
    # ==========================
    torch.save(model.state_dict(), "outputs/models/resnet18_binary_skin_fast.pth")
    print("Model saved to outputs/models/resnet18_binary_skin_fast.pth")

if __name__ == "__main__":
    main()