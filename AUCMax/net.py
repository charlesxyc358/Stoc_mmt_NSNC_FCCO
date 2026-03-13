import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys

# Define the two-layer neural network
class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        # Define the first layer (input to hidden1)
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer 1
        # Define the second layer (hidden1 to hidden2)
        self.fc2 = nn.Linear(hidden_size, input_size)  # Fully connected layer 2
        # Define the output layer (hidden2 to output)
        self.output = nn.Linear(input_size, output_size) 
        self.bn = nn.BatchNorm1d(output_size, affine=False)  # Batch Normalization for output
        
        ### initialization
        # Apply He initialization to the weights of all layers
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')  
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')  
        nn.init.kaiming_normal_(self.output.weight, nonlinearity='relu') 
        
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the second layer
        x = self.bn( self.output(x) )        # Pass through the output layer

        return x

def logger_init(out_pth):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(stream_handler)

    # FileHandler
    file_handler = logging.FileHandler(out_pth)
    file_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('logging')
    
    return logger
