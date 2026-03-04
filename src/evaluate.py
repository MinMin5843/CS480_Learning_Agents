import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, test_loader, save_path):
    """
    Calculates the classification accuracy and generates the confusion matrix for a trained model on a 
    given dataset split.

    Args:
        model: the trained neural network to be evaluated.
        test_loader: batches of samples from the split to evaluate.
        save_path: the file path where the confusion matrix image will be saved.

    Yields:
        A float containing the overall test accuracy as a fraction between 0 to 1. 
    """
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