import torch
import torch.nn as nn

class DigitNet(nn.Module):
    def __init__(self, hidden_size):
        """
        Initializes the DigitNet neural network.

        Args: 
            hidden_size: the number of neurons in the hidden layer.
        """
        super().__init__()
        self.fc1 = nn.Linear(1024, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        """
        Defines the forward pass of the model  by mapping an input batch of 
        the flattened bitmap images to class logits.

        Args:
            x: the input tensor containing the flattened 32-by-32 digit images.

        Yields:
            An output tensor containing the unnormalized class scores for each digit class. 
        """
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  