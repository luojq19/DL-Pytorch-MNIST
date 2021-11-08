# -*- codeing = utf-8 -*-
# @Time:  8:29 下午
# @Author: Jiaqi Luo
# @File: train.py
# @Software: PyCharm

import torch
import matplotlib.pyplot as plt

def train(data_train, data_loader_train, model, loss, optimizer, num_epochs, lr, device):
    avg_batch_loss = []
    avg_batch_acc = []
    for epoch in range(num_epochs):
        # training
        sum_loss = 0.0
        train_correct = 0
        for data in data_loader_train:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()

            _, id = torch.max(outputs, 1)
            sum_loss += l.item()
            train_correct += torch.sum(id == labels).item()

        avg_batch_loss.append(sum_loss / len(data_loader_train))
        avg_batch_acc.append(train_correct / len(data_train))
        print('[%d,%d] loss:%.03f' % (epoch + 1, num_epochs, sum_loss / len(data_loader_train)))
        print('        correct:%.03f%%' % (100 * train_correct / len(data_train)))

    torch.save(model.state_dict(), './models/mnist_classifier' + model.name() + str(loss)[:3] + str(optimizer)[:3] + '.pth')
    print("Trained model saved in mnist_classifier.pth")

    return model, avg_batch_loss, avg_batch_acc