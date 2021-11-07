# -*- codeing = utf-8 -*-
# @Time:  12:36 上午
# @Author: Jiaqi Luo
# @File: new.py.py
# @Software: PyCharm

import torch
import torchvision
from plot import plot_loss_and_acc

# model = torchvision.models.DenseNet()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# print(str(optimizer)[:3])
# loss = torch.nn.CrossEntropyLoss()
# print(str(loss)[:3])
# dic = {'a': 'b', 'b': 'c'}
# print(list(dic.keys())[0])
# # print(list(dic.values())[0])
# loss = [1,2,3,4]
# # acc = [[1], [2], [3], [4] , [5], [6], [7], [8], [9], [10], [11], [12]]
# acc = [0.1,0.2,0.3,0.4]
# plot_loss_and_acc({"loss": [loss, acc]})
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)