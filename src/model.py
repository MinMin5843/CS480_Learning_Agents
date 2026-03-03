import torch
import torch.nn as nn

class DigitNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(1024, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  