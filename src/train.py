import torch
import torch.nn as nn
import torch.optim as optim
from model import DigitNet

def train_model(hidden_size, train_loader, val_loader, epochs=30):
    model = DigitNet(hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                preds = model(X)
                correct += (preds.argmax(1) == y).sum().item()
                total += len(y)
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f}")

    return model