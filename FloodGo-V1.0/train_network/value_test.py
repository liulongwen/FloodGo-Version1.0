import torch
import numpy as np
import pandas as pd
import torchvision
from d2l.torch import get_dataloader_workers
from torchvision import transforms
from torch import nn
from d2l import torch as d2l
from torch.utils import data
import os
import random
from train_network.calculate_tool.data_preprocess3 import generate_dataset3
from train_network.calculate_tool.data_preprocess3 import generate_dataset_allexcel3
from train_network.calculate_tool.labels_restore import labels_restore4_true
from train_network.calculate_tool.labels_restore import labels_restore4_predict
from train_network.calculate_tool.data_preprocess1 import timefn
from train_network.calculate_tool import Z_V
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import time


class VNet(nn.Module):
    """
    test value network module
    """

    def __init__(self, in_width, in_height, out_width, out_height):
        super(VNet, self).__init__()

        self.in_width = in_width
        self.in_height = in_height
        self.out_width = out_width
        self.out_height = out_height

        # common network layer
        self.conv1 = nn.Conv2d(24, 96, kernel_size=11, stride=2, padding=5)  # first convolution layer
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)  # second convolution layer
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.flatten1 = nn.Flatten()

        # value network unique layer
        self.value_fc1 = nn.Linear(4800, 1024)  # third Fully connected layer
        self.value_dro1 = nn.Dropout(p=0.3)
        self.value_fc2 = nn.Linear(1024, 256)  # fourth Fully connected layer
        self.value_dro2 = nn.Dropout(p=0.3)
        self.value_fc3 = nn.Linear(256, 64)  # 5th Fully connected layer
        self.value_dro3 = nn.Dropout(p=0.3)
        self.value_fc4 = nn.Linear(64, 1)  # 6th Fully connected layer

    def forward(self, state_input):
        # common network layer
        x = F.relu(self.conv1(state_input.float()))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten1(x)

        # value network unique layer
        x_value = F.relu(self.value_fc1(x))
        x_value = self.value_dro1(x_value)
        x_value = F.relu(self.value_fc2(x_value))
        x_value = self.value_dro2(x_value)
        x_value = F.relu(self.value_fc3(x_value))
        x_value = self.value_dro3(x_value)
        x_value = torch.tanh(self.value_fc4(x_value))

        return x_value


np.set_printoptions(threshold=np.inf, linewidth=np.inf)
torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(suppress=True, formatter={'float_kind': '{:d}'.format})


# Create a model instance
model = VNet(60, 60, 20, 10)
X = torch.rand(size=(1, 24, 60, 60), dtype=torch.float32)
for name, layer in model.named_children():
    X = layer(X)
    print(name, 'output shape:', X.shape)
print('')


@timefn
def predict_ch3(MyModel1, features_predict1, value_labels_predict1, device,
                is_shown=True):  # features_predict, labels_predict1,
    model_new = MyModel1(60, 60, 20, 10)
    # Loading of model weights
    model_new.load_state_dict(torch.load(r'C:\APP\Python\FloodGo-V1.0\train_network\model_weight\using_best_value.model'))  # Change to your file path
    # model_new.load_state_dict(torch.load(r'./best_value.model'))
    model_new.to(device)
    torch.set_printoptions(precision=8, sci_mode=False)
    features_predict1, value_labels_predict1 = features_predict1.to(device), value_labels_predict1.to(device)
    features_predict1 = features_predict1.unsqueeze(1)
    model_new.eval()
    number_predict1 = np.zeros([5], dtype=np.int32)
    number_predict2 = np.zeros([6], dtype=np.int32)
    flow_and_Z = np.zeros((features_predict1.shape[0], 4))
    flood_value_true_test, flood_value_predict_test = [], []
    for i in range(features_predict1.shape[0]):
        # true value
        flood_value_trues = value_labels_predict1[i].item()
        # predict value
        flood_value_predicts = model_new(features_predict1[i])
        if is_shown:
            print(flood_value_predicts.shape)
        flood_value_predicts = flood_value_predicts.item()

        # print data
        if is_shown:
            print(i + 1, 'th set of data:')
            print(f'true flood control effect:{flood_value_trues:.4f}')
        flood_value_true_test.append(flood_value_trues)
        if is_shown:
            print(f'predict flood control effect:{flood_value_predicts:.4f}')
        flood_value_predict_test.append(flood_value_predicts)

        # Calculated accuracy value
        for j in range(1, 5):
            # accuracy of the statistical value network
            if flood_value_trues - j * 0.025 <= flood_value_predicts <= flood_value_trues + j * 0.025:
                number_predict2[j] = number_predict2[j] + 1

        if is_shown:
            print('')
            print("=====================================================================")
            print('')

    # print accuracy distribution
    print("value network precision results：\n")
    for temp in range(1, 5):
        predict_acc2 = number_predict2[temp] / (features_predict1.shape[0])
        print(f"The accuracy of prediction(±{temp * 0.025:.3f}):{predict_acc2:.3f}")
    print('')
    return flood_value_true_test, flood_value_predict_test


def test_dataset():
    features_predict1, policy_labels_predict1, value_labels_predict1 = \
        generate_dataset_allexcel3(r'C:\Users\Longwen-Liu\Desktop\FloodGo-Version1.0-main\data\Test flood data.xlsx')  # Change to your file path
    # Convert a matrix to a tensor
    # test dataset
    features_predict1 = torch.tensor(features_predict1, dtype=torch.float16)
    value_labels_predict1 = torch.tensor(value_labels_predict1, dtype=torch.float16)

    # Determine the hyperparameter learning rate and the number of iterations start training
    true_data1, predict_data1 = predict_ch3(VNet, features_predict1, value_labels_predict1, d2l.try_gpu(),
                                            is_shown=False)


# Read test dataset
print("Read test dataset.")
features_predict, policy_labels_predict, value_labels_predict = \
    generate_dataset_allexcel3(r'C:\Users\Longwen-Liu\Desktop\FloodGo-Version1.0-main\data\Test flood data.xlsx')  # Change to your file path
print(f"test dataset shape：{features_predict.shape}")

# Convert a matrix to a tensor
# test dataset
features_predict = torch.tensor(features_predict, dtype=torch.float16)
value_labels_predict = torch.tensor(value_labels_predict, dtype=torch.float16)

# Determine the hyperparameter learning rate and the number of iterations start training
true_data, predict_data = predict_ch3(VNet, features_predict, value_labels_predict, d2l.try_gpu())

# specify the file path to save
file_path = r'C:\Users\Longwen-Liu\Desktop\data\value.txt'  # Change to your file path
with open(file_path, 'w') as file:
    file.truncate(0)
for true, predict in zip(true_data, predict_data):
    loss_txt = f'{true:.4f} {predict:.4f}\n'
    with open(file_path, 'a') as file:
        file.write(loss_txt)
print("Data write success!")
