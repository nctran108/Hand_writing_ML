import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

class mnistNN(nn.Module):
    def __init__(self,in_channels: int,hidden_channels: int,output_channels: int,num_features: int, subSampling: int, kernal_size: int = 28):
        super(mnistNN, self).__init__()
        # kernel: 1 input image in 2D, num_features output channels
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=num_features,kernel_size=kernal_size)
        self.conv2 = nn.Conv2d(in_channels=num_features,out_channels=subSampling, kernel_size=kernal_size)

        self.fc1 = nn.Linear(subSampling * (kernal_size-1)**2, out_features=hidden_channels)
        self.fc2 = nn.Linear(in_features=hidden_channels,out_features=50)
        self.fc3 = nn.Linear(in_features=50,out_features=output_channels)



    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # print(x.shape)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))

        return x

def main():
    mnist = mnistNN(1,120,10,6,16,5)
    print(mnist)
    params = list(mnist.parameters())
    print(len(params))
    print(params[0].size())

if __name__ == "__main__":
    main()