# -*- codeing = utf-8 -*-
# @Time:  8:17 下午
# @Author: Jiaqi Luo
# @File: mnist_classifiers.py
# @Software: PyCharm

import torch
from torch import nn
import torchvision
from torchvision import transforms
from train import train
from test import test
from plot import plot_loss_and_acc

batch_size = 64

# Load and process the mnist dataset with torchvision
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.5], std=[0.5])])

data_train = torchvision.datasets.MNIST(root="./data/",
                                        transform=transform,
                                        train=True,
                                        download=True)

data_test = torchvision.datasets.MNIST(root="./data/",
                                       transform=transform,
                                       train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=batch_size,
                                                shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=batch_size,
                                               shuffle=True)

class Model1(nn.Module):  # ConvNet
    # 3*3*64 cov + 3*3*128 cov + 2*2 maxpool + fc + fc
    def __init__(self):
        super(Model1, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(stride=2, kernel_size=2))
        self.fc = nn.Sequential(nn.Linear(14 * 14 * 128, 1024),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(1024, 10))

    def forward(self, x):
        return self.fc(self.conv(x).view(-1, 14 * 14 * 128))

    def name(self):
        return "Model1"

class Model2(nn.Module):  # ConvNet
    # use kernel size of 2
    def __init__(self):
        super(Model2, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(stride=2, kernel_size=2))
        self.fc = nn.Sequential(nn.Linear(15 * 15 * 128, 1024),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(1024, 10))

    def forward(self, x):
        return self.fc(self.conv(x).view(-1, 15 * 15 * 128))

    def name(self):
        return "Model2"

class Model3(nn.Module):  # MLP with 2 layers
    def __init__(self):
        super(Model3, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(784, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 10))

    def forward(self, x):
        return self.mlp(x.view(-1, 784))

    def name(self):
        return "Model3"

class Model4(nn.Module):  # MLP with 3 layers
    def __init__(self):
        super(Model4, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(784, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(256, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 10))

    def forward(self, x):
        return self.mlp(x.view(-1, 784))

    def name(self):
        return "Model4"

class Model5(nn.Module):  # MLP with 2 layers with sigmoid
    def __init__(self):
        super(Model5, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(784, 64),
                                 nn.Sigmoid(),
                                 nn.Linear(64, 10))

    def forward(self, x):
        return self.mlp(x.view(-1, 784))

    def name(self):
        return "Model5"

class Model6(nn.Module):  # ConvNet with very few channels
    def __init__(self):
        super(Model6, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(stride=2, kernel_size=2))
        self.fc = nn.Sequential(nn.Linear(14 * 14 * 6, 128),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(128, 10))

    def forward(self, x):
        return self.fc(self.conv(x).view(-1, 14 * 14 * 6))

    def name(self):
        return "Model6"

lr = 0.001
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Start training on ", device)


# a test case for the train, test and plot functions
'''
model = Model3()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print('#' * 40)
print("Begin train and test: " + model.name())
print(str(optimizer)[:3])
print(str(loss)[:3])

model.to(device)
# train
model, batch_loss, batch_acc = train(data_train, data_loader_train, model, loss, optimizer, num_epochs, lr, device)

# plot
plot_loss_and_acc({model.name() + "+" + str(optimizer)[:3] + "+" + str(loss)[:3]: [batch_loss, batch_acc]})

# test
accuracy = test(model, data_loader_test)
# i += 1
print('#' * 40)
print('\n')
'''

# i = 0

for model in [Model1(), Model2(), Model3(), Model4(), Model5()]:
    for loss in [nn.CrossEntropyLoss(), nn.MSELoss()]:
        for optimizer in [torch.optim.Adam(model.parameters(), lr=lr),
                          torch.optim.SGD(model.parameters(), lr=lr)]:
            print('#' * 40)
            print("Begin train and test: " + model.name())
            print(str(optimizer)[:3])
            print(str(loss)[:3])

            model.to(device)
            # train
            model, batch_loss, batch_acc = train(data_train, data_loader_train, model, loss, optimizer, num_epochs, lr, device)

            # plot
            plot_loss_and_acc({model.name() + "+" + str(optimizer)[:3] + "+" + str(loss)[:3]: [batch_loss, batch_acc]})

            # test
            accuracy = test(model, data_loader_test)
            # i += 1
            print('#' * 40)
            print('\n')

print("Complete training and testing on ", device)
