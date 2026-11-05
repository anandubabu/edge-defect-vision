"""Final evaluation: run the chosen model once on the unseen test set."""
import torch
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

from dataset import DefectDataset

MODEL_PATH = "models/best_lr0.0003.pth"
TEST_CSV = "data/split/test.csv"


def main():
    # Class names in the same sorted order used for the labels 0-5.
    test_df = pd.read_csv(TEST_CSV)
    classes = sorted(test_df["class"].unique())

    # Rebuild the same architecture as in training, then load the trained weights.
    # weights=None: no ImageNet download, our saved weights replace them anyway.
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 6)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # testing mode

    # Test images, prepared exactly like validation (no random augmentation).
    # shuffle=False keeps the CSV order, so predictions can be matched to file paths.
    test_ds = DefectDataset(TEST_CSV, train=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    # Predict every test image and keep the results.
    all_labels = []
    all_preds = []
    with torch.no_grad():  # no learning here, just predicting
        for images, labels in test_loader:
            preds = model(images).argmax(dim=1)  # highest of the 6 scores = predicted class
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())

    # Overall score and per-class precision / recall.
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test accuracy: {acc:.4f} ({len(all_labels)} images)")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=3))

    # Confusion matrix: rows = true class, columns = predicted class.
    ConfusionMatrixDisplay.from_predictions(
        all_labels, all_preds, display_labels=classes, xticks_rotation=45)
    plt.title("Confusion matrix - test set")
    plt.savefig("docs/confusion_matrix.png", bbox_inches="tight")
    print("saved docs/confusion_matrix.png")

    # List every wrong prediction, for error analysis.
    mistakes = 0
    for i, (true, pred) in enumerate(zip(all_labels, all_preds)):
        if true != pred:
            mistakes += 1
            print(f"WRONG: {test_df.loc[i, 'path']}  true={classes[true]}  predicted={classes[pred]}")
    print(f"{mistakes} wrong out of {len(all_labels)}")


if __name__ == "__main__":
    main()