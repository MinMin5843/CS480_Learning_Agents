import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, test_loader, save_path):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            preds = model(X).argmax(1)
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    acc = accuracy_score(y_true, y_pred)
    print("Test Accuracy:", acc)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.savefig(save_path)
    plt.close()

    return acc