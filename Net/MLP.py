import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_sizes=[128, 64], output_size=10):
        """
        Initializes an MLP with customizable layer sizes.

        Args:
            input_size (int): Number of input features (default is 28*28 for MNIST).
            hidden_sizes (list): List containing the number of units in each hidden layer.
            output_size (int): Number of output features (default is 10 for MNIST digit classification).
        """
        super(MLP, self).__init__()
        
        # Create a list of fully connected layers
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        
        # Last hidden layer to output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Use nn.Sequential to stack the layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the input image
        x = x.view(x.size(0), -1)
        # Forward pass through the network
        return self.network(x)
