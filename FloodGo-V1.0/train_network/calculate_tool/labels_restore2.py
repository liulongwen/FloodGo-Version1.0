from d2l import torch as d2l
import pandas as pd
import numpy as np
import torch


def labels_restore4_true(features, count, device):
    features = features.to(device)
    m = int((count + 1) // (80 / 2))
    n = int((count + 1) % (80 / 2))
    values, indices = torch.max(features[count, 0, 1:5, m * 10:(m + 1) * 10, n * 2 - 1], dim=1)
    out_flood_data = int(indices[0] * 1000 + indices[1] * 100 + indices[2] * 10 + indices[3])
    return out_flood_data
    # return out_flood_data, indices


def labels_restore4_predict(labels, device):
    labels = labels.to(device)
    labels = labels.to(torch.float32)
    labels = labels.round(decimals=8)
    indices = d2l.argmax(labels, axis=1)
    out_flood_data = (int(indices) + 1) * 25
    return out_flood_data
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
# b = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
# c = torch.tensor([11])
# a = d2l.accuracy(b, c)
# print(a)
# print(-1//25)
