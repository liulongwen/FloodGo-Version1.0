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


# policy network structure
class PNet(nn.Module):
    """
    test policy network module
    """

    def __init__(self, in_width, in_height, out_width, out_height):
        super(PNet, self).__init__()

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

        # policy network unique layer
        self.policy_fc1 = nn.Linear(4800, 1024)  # third Fully connected layer
        self.policy_dro1 = nn.Dropout(p=0.3)
        self.policy_fc4 = nn.Linear(1024, self.out_width * self.out_height)  # fourth Fully connected layer

    def forward(self, state_input):
        # common network layer
        x = F.relu(self.conv1(state_input.float()))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten1(x)

        # policy network unique layer
        x_policy = F.relu(self.policy_fc1(x))
        x_policy = self.policy_dro1(x_policy)
        x_policy = self.policy_fc4(x_policy)

        return x_policy


np.set_printoptions(threshold=np.inf, linewidth=np.inf)
torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(suppress=True, formatter={'float_kind': '{:d}'.format})

# Create a model instance
model = PNet(60, 60, 20, 10)

X = torch.rand(size=(1, 24, 60, 60), dtype=torch.float32)

for name, layer in model.named_children():
    X = layer(X)
    print(name, 'output shape:', X.shape)
print('')

# Read test dataset
inflow_true, Z_up_true, Z_down_true = [], [], []
trues_test, predicts_test, labels_trues_test, labels_predicts_test = [], [], [], []
print("read test dataset")
# single flood forecast
# features_predict, policy_labels_predict, value_labels_predict, inflow_true, Z_up_true, Z_down_true = generate_dataset_allexcel3(
#     r'C:\Users\Longwen-Liu\Desktop\洪水预处理\预测洪水数据-5.xlsx', labels_preprocess=False)
# multiple flood forecasts
features_predict, policy_labels_predict, value_labels_predict = generate_dataset_allexcel3(
    r'C:\Users\Longwen-Liu\Desktop\FloodGo-Version1.0-main\data\Test flood data.xlsx')  # Change to your file path
print(f"test dataset shape：{features_predict.shape}")

# Define batch number
batch_size = 2048

# Convert a matrix to a tensor
# test dataset
features_predict = torch.tensor(features_predict, dtype=torch.float16)
policy_labels_predict = torch.tensor(policy_labels_predict, dtype=torch.long)
value_labels_predict = torch.tensor(value_labels_predict, dtype=torch.float16)


@timefn
def predict_ch3(MyModel1, features_predict1, policy_labels_predict1, device, is_flood=False,
                labels_preprocess=True):  # features_predict, labels_predict1,
    model_new = MyModel1(60, 60, 20, 10)
    # Loading of model weights
    model_new.load_state_dict(torch.load(r'C:\APP\Python\FloodGo-V1.0\train_network\model_weight\using_best_policy.model'))  # Change to your file path
    # model_new.load_state_dict(torch.load(r'current_policy.model'))
    # model_new.load_state_dict(torch.load(r'best_policy.model'))
    model_new.to(device)
    features_predict1, policy_labels_predict1 = features_predict1.to(device), policy_labels_predict1.to(device)
    features_predict1 = features_predict1.unsqueeze(1)
    model_new.eval()
    number_predict1 = np.zeros([5], dtype=np.int32)
    flow_and_Z = np.zeros((features_predict1.shape[0], 4))
    for i in range(features_predict1.shape[0]):
        # true value
        trues, labels_trues = labels_restore4_true(policy_labels_predict1[i], d2l.try_gpu(),
                                                   labels_preprocess=labels_preprocess)
        # predict value
        labels_probs = model_new(features_predict1[i])
        predicts, labels_predicts = labels_restore4_predict(labels_probs, d2l.try_gpu())

        # Store inflow and outflow data
        if is_flood:
            if i == 0:
                Z_up = round(Z_up_true[0], 2)
                Z_down = round(Z_down_true[0], 2)
            else:
                Z0 = Z_up_true[0]
                diff_Z = Z_up_true[0] - Z_down_true[0]
                Z_up, Z_down = Z_V.calculate_Z_up_down(inflow_true[i], predicts, Z0, diff_Z)
            flow_and_Z[i, 0] = inflow_true[i]
            flow_and_Z[i, 1] = predicts
            flow_and_Z[i, 2] = Z_up
            flow_and_Z[i, 3] = Z_down

        # print data
        print(i + 1, 'th set of data:')
        print('true outflow label:', labels_trues)
        labels_trues_test.append(labels_trues)
        print('true outflow:', trues)
        trues_test.append(trues)
        print('predict outflow label:', labels_predicts)
        labels_predicts_test.append(labels_predicts)
        print('predict outflow:', predicts)
        predicts_test.append(predicts)

        # Calculated accuracy value
        for j in range(5):
            if int(labels_trues) - j * 1 <= labels_predicts <= int(labels_trues) + j * 1:
                number_predict1[j] = number_predict1[j] + 1

        print('')
        print("=====================================================================")
        print('')

    # print accuracy distribution
    print("policy network precision results：\n")
    for temp in range(5):
        predict_acc1 = number_predict1[temp] / (features_predict1.shape[0])
        print(f"The accuracy of prediction (±{temp * 50}):{predict_acc1:.3f}")
    print('')

    if is_flood:
        # The scheduling result is output
        with open('policy_result.txt', 'w') as file:
            file.truncate(0)
        # Writes data to a file
        with open('policy_result.txt', 'w') as file:
            for i in range(len(flow_and_Z[:, 0])):
                result_txt = f'{flow_and_Z[i, 0]}, {flow_and_Z[i, 1]}, {flow_and_Z[i, 2]}, {flow_and_Z[i, 3]}\n'
                file.write(result_txt)
        file.close()
        # print display results
        print("inflow：", flow_and_Z[:, 0])
        print("outflow：", flow_and_Z[:, 1])
        print("water level above the dam：", flow_and_Z[:, 2])
        print("water level below the dam：", flow_and_Z[:, 3], "\n")
    return trues_test, predicts_test, labels_trues_test, labels_predicts_test


# confusion matrix function to observe the data distribution of the validation set
def confusion_matrix(true_data1, predict_data1):
    matrix1 = np.zeros((10, 10))
    for index, (true, predict) in enumerate(zip(true_data1, predict_data1)):
        x, y = None, None
        for i in range(10):
            if i * 20 <= true < (i + 1) * 20:
                y = i
            if i * 20 <= predict < (i + 1) * 20:
                x = 9 - i
        matrix1[x, y] += 1
    return matrix1


# determine the hyperparameter learning rate and the number of iterations to start training
trues_data, predicts_data, trues_labels_data, predicts_labels_data = \
    predict_ch3(PNet, features_predict, policy_labels_predict, d2l.try_gpu(), is_flood=False, labels_preprocess=True)

print(trues_labels_data)
print(predicts_labels_data)
print(max(trues_labels_data))
print(max(predicts_labels_data))
matrix = confusion_matrix(trues_labels_data, predicts_labels_data)
print(matrix)

# Specify the file path to save
file_path = r'C:\Users\Longwen-Liu\Desktop\data\policy.txt'  # Change to your file path

with open(file_path, 'w') as file:
    file.truncate(0)
for true, predict in zip(trues_data, predicts_data):
    loss_txt = f'{true} {predict}\n'
    with open(file_path, 'a') as file:
        file.write(loss_txt)
print("Data write success!")
