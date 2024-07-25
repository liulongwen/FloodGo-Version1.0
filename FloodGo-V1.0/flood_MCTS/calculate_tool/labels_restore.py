from d2l import torch as d2l
import pandas as pd
import numpy as np
import torch


def labels_restore4_true(labels, device):
    labels = labels.to(device)
    trues = int(labels.item())
    temp = round(trues / 25) - 1
    if temp == -1:
        temp = 0
    labels_trues = temp
    return trues, labels_trues


def labels_restore4_predict(labels, device):
    labels = labels.to(device)
    labels = labels.to(torch.float32)
    labels = labels.round(decimals=8)
    indices = int(d2l.argmax(labels, axis=0))
    out_flood_data = (indices + 1) * 25
    return out_flood_data, indices
    # return out_flood_data, indices

# Example usage
# data = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         1, 1, 2, 0, 0, 0, 0, 0, 0, 0,
#         0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
# label_test = torch.tensor(data).reshape(1, 30)
# output, indice = labels_restore4_predict(label_test, d2l.try_gpu())
# print(label_test)
# print('subscript matrix:\n', indice)
# print('outflow:', output)

# data = [600, 600, 711, 749, 749, 750, 750, 751, 752, 752]
# label_test = torch.tensor(data).reshape(10, 1)
# trues, labels_trues = labels_restore4_true(label_test[0], d2l.try_gpu())
# print(label_test)
# print('true inflow:\n', trues)
# print('true inflow label:', labels_trues)

# b = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
# c = torch.tensor([11])
# a = d2l.accuracy(b, c)
# print(a)
# print(-1//25)
