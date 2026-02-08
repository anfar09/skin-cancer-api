import torch
import matplotlib.pyplot as plt

stats = torch.load("outputs/figures/training_stats.pth")

train_losses = stats["train_losses"]
val_losses = stats["val_losses"]

epochs = range(1, len(train_losses) + 1)

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss (Overfitting Check)")
plt.legend()
plt.grid()

plt.savefig("outputs/figures/overfitting_curve.png")
plt.show()