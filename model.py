import torch
# Import neural network module from PyTorch
import torch.nn as nn

# Define a custom neural network class by inheriting PyTorch's nn.Module
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initialize neural network with three layers
        Parameters:
        - input_size: Number of input features (size of input layer)
        - hidden_size: Number of neurons in the hidden layers
        - num_classes: Number of output classes (size of output layer)
        """

        # Call the parent class nn.Module
        super(NeuralNet, self).__init__()

        # Define the first layer: input -> hidden
        self.l1 = nn.Linear(input_size, hidden_size)
        # Define the second layer: hidden -> hidden
        self.l2 = nn.Linear(hidden_size, hidden_size)
        # Define the third layer: hidden -> output
        self.l3 = nn.Linear(hidden_size, num_classes)
        # Define the activation function (ReLU) to introduce non-linearity
        # helps the model understand complex patterns in data
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        Parameters:
        - x: Input tensor (batch of data)
        Returns:
        - Output tensor (raw scores/logits for each class, not probabilities)
        """

        # Pass input through the first layer
        out = self.l1(x)
        # Apply ReLU activation function
        out = self.relu(out)
        # Pass through the second layer
        out = self.l2(out)
        # Apply ReLU again
        out = self.relu(out)
        # Pass through the final layer
        out = self.l3(out)
        # No activation (like softmax) at the end because:
        # - If using CrossEntropyLoss, PyTorch applies softmax automatically.
        # - This allows the model to output raw scores (logits).
        return out
