from src.dataset import load_data
from src.train import train_model
from src.evaluate import evaluate
import torch

train_loader, val_loader, test_loader = load_data()

for hidden in [10, 100, 500]:
    print(f"\nTraining model with {hidden} hidden nodes...")
    model = train_model(hidden, train_loader, val_loader)

    torch.save(model.state_dict(), f"models/model_{hidden}.pth")

    acc = evaluate(model, test_loader, f"outputs/confusion_{hidden}.png")

    with open(f"outputs/accuracy_{hidden}.txt", "w") as f:
        f.write(f"Test Accuracy: {acc}\n")