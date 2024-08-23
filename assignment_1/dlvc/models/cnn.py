from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch




class YourCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # initialize the layers
        # 2 convolutional layers
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size = 3, padding = 1) # take 3 channels as input
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size = 3, padding = 1) # output 32 channels
        self.pool = nn.MaxPool2d(2, 2) # apply max pooling with kernel size 2 and stride 2 
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        # define the forward pass
        # apply relu after each convolution
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  
        return x

class YourCNNWithDropout(YourCNN):
    def __init__(self, dropout_prob = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout_prob)

    def forward(self, x):
        # define the forward pass
        # apply relu after each convolution
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # dropout
        x = self.fc2(x)  
        return x