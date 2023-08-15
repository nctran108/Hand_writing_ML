import numpy as np
import pandas as pd
from modules import mnistNN
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim

n_epochs = 3
batch_size_train = 60
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.9
log_interval = 10

random_seed = 1
torch.manual_seed(random_seed)

train_data = pd.read_csv('mnist/mnist_train.csv')
row, colm = train_data.shape
sample_data = train_data.sample(frac=1)

train_label = torch.tensor(sample_data["label"]).reshape(int(row/batch_size_train),batch_size_train,1).cuda()
print(train_label.shape)
train_image = torch.tensor(sample_data[train_data.columns[1:]].to_numpy())

test_data = pd.read_csv('mnist/mnist_test.csv')
row, colm = test_data.shape
sample_data = test_data.sample(frac=1)

test_label = torch.tensor(sample_data["label"]).reshape(int(row/batch_size_test),batch_size_test,1).cuda()
test_image = torch.tensor(sample_data[train_data.columns[1:]].to_numpy())

mnist = mnistNN(1,320,10,10,20,5).cuda()
print(mnist)
params = list(mnist.parameters())
print(len(params))
print(params[0].size())

train_loader = train_image.reshape(int(len(train_image)/batch_size_train),batch_size_train,1,28,28).type(torch.float).cuda()
test_loader = test_image.reshape(int(len(test_image)/batch_size_test),batch_size_test,1,28,28).type(torch.float).cuda()

print(train_loader.shape)
print(train_loader[0].shape)

output = mnist(train_loader[0])

#img = F.interpolate(img, size=32)

print(output)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mnist.parameters(), lr=0.001, momentum=0.9)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_image) for i in range(n_epochs + 1)]

def train(epoch):
    mnist.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        target = train_label[batch_idx].flatten()
        output = mnist(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data) , len(train_image),
                100. * batch_idx / len(train_loader), loss.item(),))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*60)+((epoch-1)*len(train_loader)))
            torch.save(mnist.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')


def test():
    mnist.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            output = mnist(data)
            target = test_label[batch_idx].flatten()
            test_loss += F.nll_loss(output, target, size_average=False).item()

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        
        test_loss /= len(test_image)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_image), 100. * correct / len(test_image)
        ))

test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()