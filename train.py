"""Train, select by validation loss, then evaluate the selected checkpoint once."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import DigitCNN

OUT = Path(__file__).resolve().parent / "results"
SEED = 42


def augment(images):
    """Independently rotate by up to 10 degrees and zoom by up to 8%."""
    batch = len(images)
    angle = (torch.rand(batch) * 2 - 1) * math.radians(10)
    zoom = 0.92 + 0.16 * torch.rand(batch)
    cosine, sine = angle.cos() / zoom, angle.sin() / zoom
    zero = torch.zeros_like(angle)
    affine = torch.stack((cosine, -sine, zero, sine, cosine, zero), dim=1)
    grid = F.affine_grid(affine.reshape(-1, 2, 3), images.shape, align_corners=False)
    return F.grid_sample(images, grid, align_corners=False)


def save_rows(filename, header, rows):
    with (OUT / filename).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = load_digits()
    images = torch.tensor(data.images, dtype=torch.float32).unsqueeze(1) / 16
    labels = torch.tensor(data.target, dtype=torch.long)
    development, test = train_test_split(np.arange(len(labels)), test_size=0.15,
                                         stratify=data.target, random_state=SEED)
    train, valid = train_test_split(development, test_size=len(test),
                                    stratify=data.target[development], random_state=SEED)
    loader = DataLoader(TensorDataset(images[train], labels[train]), batch_size=64, shuffle=True)
    model = DigitCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    history = []
    best_loss, best_epoch = float("inf"), 0
    max_epochs, patience = 75, 10
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0
        for batch, truth in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(augment(batch)), truth)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.inference_mode():
            logits = model(images[valid])
            valid_loss = F.cross_entropy(logits, labels[valid]).item()
            valid_acc = (logits.argmax(1) == labels[valid]).float().mean().item()
        history.append((epoch, total / len(train), valid_loss, valid_acc))
        if valid_loss < best_loss:
            best_loss, best_epoch = valid_loss, epoch
            torch.save(model.state_dict(), OUT / "best_model.pt")
        if epoch >= 10 and epoch - best_epoch >= patience:
            break

    model.load_state_dict(torch.load(OUT / "best_model.pt", weights_only=True))
    with torch.inference_mode():
        predicted = model(images[test]).argmax(1).numpy()
    truth = data.target[test]
    report = classification_report(truth, predicted, labels=np.arange(10),
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=np.arange(10))
    save_rows("test_predictions.csv", ["dataset_index", "actual", "predicted"],
              zip(test, truth, predicted))
    save_rows("history.csv", ["epoch", "train_loss", "validation_loss", "validation_accuracy"], history)
    save_rows("confusion_matrix.csv", ["actual/predicted", *range(10)],
              ([i, *row] for i, row in enumerate(matrix)))
    metrics = ["precision", "recall", "f1-score", "support"]
    save_rows("classification_report.csv", ["class", *metrics],
              ([k, *(v[m] for m in metrics)] for k, v in report.items() if isinstance(v, dict)))
    summary = dict(train=len(train), validation=len(valid), test=len(test), epochs=epoch,
                   best_epoch=best_epoch, best_validation_loss=best_loss, patience=patience,
                   stopped_early=epoch < max_epochs, report=report)
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    figure, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=range(10)).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Handwritten digit recognition · {len(test)} held-out images")
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("Actual digit")
    figure.tight_layout()
    figure.savefig(OUT / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    epochs, train_loss, val_loss, _ = zip(*history)
    figure, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="Training (augmented)")
    ax.plot(epochs, val_loss, label="Validation")
    ax.axvline(best_epoch, color="gray", linestyle=":", label="Selected epoch")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training and validation loss")
    ax.legend()
    figure.tight_layout()
    figure.savefig(OUT / "learning_curve.png", dpi=160)
    plt.close(figure)
    print(json.dumps({k:v for k,v in summary.items() if k != "report"}))
    print(classification_report(truth, predicted, digits=4))


if __name__ == "__main__":
    main()
