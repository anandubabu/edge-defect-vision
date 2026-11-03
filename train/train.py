import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from pathlib import Path
import mlflow

from dataset import DefectDataset

def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    total_loss=0
    for image ,labels in train_loader:
        outputs=model(image)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(train_loader)

def evaluate (model, loader):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for image, labels in loader:
            outputs=model(image)
            pred=outputs.argmax(dim=1)
            correct += (pred==labels).sum().item()
            total += labels.size(0)
    return correct/total

def main():
    lr=0.0003
    batch_size=32
    epochs=10

    train_ds = DefectDataset("data/split/train.csv", train=True)
    val_ds = DefectDataset("data/split/val.csv", train=False)

    train_loader =DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    #image, labels =next(iter(train_loader))
    #print(image.shape, labels[:8])

    model = models.mobilenet_v3_small(weights="DEFAULT")
    print(model.classifier)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 6)
    #out =model(image)
    #print(out.shape)

    loss_fn =nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    Path("models").mkdir(exist_ok=True)
    with mlflow.start_run():
        mlflow.log_param("model", "mobilenet_v3_small")
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        best_acc = 0
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
            val_acc = evaluate(model, val_loader)
            print(f"epoch {epoch + 1}: loss {train_loss:.3f}, val acc {val_acc:.3f}")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"models/best_lr{lr}.pth")

        mlflow.log_metric("best_val_acc", best_acc)
        print("best val accuracy:", best_acc)

if __name__=="__main__":
    main()