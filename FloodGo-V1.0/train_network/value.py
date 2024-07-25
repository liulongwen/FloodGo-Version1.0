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
from train_network.calculate_tool.data_preprocess3 import generate_dataset_allexcel3, generate_dataset_allexcel3_beta
from train_network.calculate_tool.file_path import generate_dataset_from_sheet, get_sheet_name, \
    generate_dataset_from_sheet_bata, generate_dataset_simulate
from train_network.calculate_tool.labels_restore import labels_restore4_true
from train_network.calculate_tool.labels_restore import labels_restore4_predict
from train_network.calculate_tool.data_preprocess1 import timefn
from train_network.calculate_tool import Z_V
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
# from train_network.value_test import test_dataset


class VNet(nn.Module):
    """
    train value network module
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

# Create a model instance
model = VNet(60, 60, 20, 10)
X = torch.rand(size=(1, 24, 60, 60), dtype=torch.float32)
for name, layer in model.named_children():
    X = layer(X)
    print(name, 'output shape:', X.shape)
print(X)
print('')


def evaluate_accuracy_gpu(model1, data_iter, device=None):  # @save
    """
    Calculate the accuracy of the model on the data set using the GPU
    """
    if isinstance(model1, nn.Module):
        model1.eval()  # Set to evaluation mode
        if not device:
            device = next(iter(model1.parameters())).device
    metric = d2l.Accumulator(3)
    with torch.no_grad():
        for X, y1 in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y1 = y1.to(device)

            # Calculate the test set loss function and precision
            with autocast():  # Semi-precision training
                loss1 = nn.MSELoss()
                y1_hat = model1(X)
                y1 = y1.unsqueeze(1)
                l1 = loss1(y1_hat, y1)
                loss = l1
                metric.add(loss * y1.numel(), value_accuracy(y1_hat, y1), y1.numel())

        loss = metric[0] / metric[2]
        accuracy = metric[1] / metric[2]

    return loss, accuracy


def value_accuracy(y_hat, y):
    """
    Compute the number of correct predictions.
    Defined in :numref:`sec_utils`
    """
    global cmp
    diff = abs(d2l.astype(y_hat, y.dtype) - y)
    cmp = diff <= 0.05
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))


# @timefn
def load_array(data_arrays, batch_size1, is_train=True):  # @save
    """
    Construct a PyTorch data iterator
    """
    # Verify that all elements are tensors and have the same size of the first dimension
    assert all(torch.is_tensor(arr) for arr in data_arrays), "Not all elements are tensors"
    assert all(data_arrays[0].size(0) == arr.size(0) for arr in data_arrays), "The sizes of tensors do not match"

    # Prints the type and shape of the tuple
    print("Data arrays:", type(data_arrays), [arr.shape for arr in data_arrays])

    # Package the data array as a TensorDataset object
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size1, shuffle=is_train)


# @timefn
def load_data_train_test(data_numpy, test_number1, isfeatures):
    indices = np.linspace(0, data_numpy.shape[0] - 1, test_number1, dtype=int)
    if isfeatures:
        test = data_numpy[indices, :, :, :]
    else:
        test = data_numpy[indices]
    keep_indices = np.ones(len(data_numpy), dtype=bool)
    keep_indices[indices] = False
    train = data_numpy[keep_indices]
    return train, test


@timefn
def load_data_train_test_predict(data_numpy1, data_numpy2, test_number1, predict_number1, isfeatures):
    # partition data set
    train1, test_predict1 = load_data_train_test(data_numpy1, test_number1, isfeatures)
    test1, predict1 = load_data_train_test(test_predict1, predict_number1, isfeatures)
    # partition data set
    train2, test_predict2 = load_data_train_test(data_numpy2, test_number1, isfeatures)
    test2, predict2 = load_data_train_test(test_predict2, predict_number1, isfeatures)
    # assembly data set
    train = np.concatenate((train1, train2), axis=0)
    validation = np.concatenate((test1, test2), axis=0)
    test = np.concatenate((predict1, predict2), axis=0)
    # Check the shape of the training set, Validation set, and validation set
    print(f"train dataset shape：{train.shape}")
    print(f"validation dataset shape：{validation.shape}")
    print(f"test dataset shape：{test.shape}")
    return train, validation, test


# training function train_ch6
# @timefn
def train_ch6(MyModule1, train_iter1, test_iter1, num_epochs1, lr1, device, is_real_train=True):
    global file_path, test_acc
    print("Training begins.")
    model_new = MyModule1(60, 60, 20, 10)

    # Initial weight
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    model_new.apply(init_weights)

    # Transfer data to GPU memory
    print('training on', device)
    model_new.to(device)
    # select the gradient optimizer
    # optimizer = torch.optim.SGD(model_new.parameters(), lr=lr1)
    optimizer = torch.optim.Adam(model_new.parameters(), lr=lr1, weight_decay=0.01)  # weight_decay=1e-2

    # select the loss function type
    loss1 = nn.MSELoss()

    # Enter the iteration loop and start training
    number = 0
    number1 = 0
    best_train_l = np.inf
    val_acc1 = 0
    for epoch in range(num_epochs1):
        # The sum of training loss, the sum of training accuracy and the number of samples were collected
        metric = d2l.Accumulator(3)
        model_new.train()
        for i, (X1, y1) in enumerate(train_iter1):
            number = number + 1
            X1, y1 = X1.to(device), y1.to(device)
            y1 = y1.unsqueeze(1)
            with autocast():
                optimizer.zero_grad()
                y1_hat = model_new(X1)
                value_loss = loss1(y1_hat, y1)
                loss = value_loss
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(loss * X1.shape[0], value_accuracy(y1_hat, y1), X1.shape[0])
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
        if is_real_train:
            print(f'epoch {epoch + 1}, train loss1 {train_l:.6f}, train acc {train_acc:.3f}')
        else:
            test_l, test_acc = evaluate_accuracy_gpu(model_new, test_iter1)
            print(f'epoch {epoch + 1}, train loss1 {train_l:.6f}, train acc {train_acc:.3f}, test loss1 {test_l:.6f},'
                  f' test acc {test_acc:.3f}')
            # specify the file path to save
            file_path = r'C:\Users\Longwen-Liu\Desktop\data\value训练数据.txt'
            # converts a tensor to a string
            loss_txt = f'epoch {epoch + 1}, train loss1 {train_l:.6f}, train acc {train_acc:.3f}, test loss1 {test_l:.6f},' \
                       f' test acc {test_acc:.3f}\n'
            # writes the tensor string to a file
            with open(file_path, 'a') as file:
                file.write(loss_txt)

        # Saves the average accuracy on the Validation dataset
        if not is_real_train:
            if (epoch + 1) == num_epochs1:
                val_acc1 = test_acc
                print('')

        # Determine whether to save the weight value
        if is_real_train:
            if (epoch + 1) % 50 == 0:
                print("Save the current model weight.")
                torch.save(model_new.state_dict(), r'current_value.model')  # Save the parameter file for the current model
                # 比较当前模型与最佳模型的胜率，择优保存
                if train_l < best_train_l or abs(train_l > best_train_l) < 0.05:
                    print("A better value network has emerged!\n")
                    best_train_l = train_l
                    torch.save(model_new.state_dict(), r'best_value.model')  # Save the parameter file of the optimal model
                    number1 = number1 + 1
                # test_dataset()

            # Specify the file path to save
            file_path = r'C:\Users\Longwen-Liu\Desktop\洪水预处理\value_loss.txt'
            # converts a tensor to a string
            loss_txt = f'epoch,{epoch + 1},train loss1,{train_l:.6f}\n'
            # writes the tensor string to a file
            with open(file_path, 'a') as file:
                file.write(loss_txt)
    if is_real_train:
        txt = f'\n'
        with open(file_path, 'a') as file:
            file.write(txt)
        print("training times：" + str(number))
        print("the number of times a model weight is saved：" + str(number1) + '\n')
    return val_acc1


# Define batch number
batch_size = 2048

# Determine the hyperparameter learning rate and the number of iterations to start training
lr, num_epochs = 0.000004, 500

# The data set is divided using K-fold cross-validation
val_acc_sum = 0
val_acc_avg = 0
val_acc_list = []
print(f'lr={lr}, num_epochs={num_epochs}, batch_size={batch_size}')
print(f'p=0.3/0.3/0.3, weight_decay={0.01}, n_splits={5}, random_state={48}, diff={0.05}\n')
print("The data set is divided using K-fold cross-validation")

# Gets all sheets names
sheet_name = get_sheet_name()

X = sheet_name

# Define k-fold cross-validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=48)

# Perform K-fold cross-validation
for fold, (train_index, val_index) in enumerate(k_fold.split(X)):
    print(f"Fold {fold + 1}:")
    print(train_index)
    print(val_index)
    sheet_name_train, sheet_name_val = [X[i] for i in train_index], [X[i] for i in val_index]

    # Divide historical data
    print("Read Excel data and convert it to a matrix.")
    # train dataset
    X_train, y_train = generate_dataset_from_sheet_bata(sheet_name_val, is_policy=False)
    # validation dataset
    X_val, y_val = generate_dataset_from_sheet(sheet_name_val, is_policy=False)

    # Convert a matrix to a tensor
    # train dataset
    X_train = torch.tensor(X_train, dtype=torch.float16)
    y_train = torch.tensor(y_train, dtype=torch.float16)
    # validation dataset
    X_val = torch.tensor(X_val, dtype=torch.float16)
    y_val = torch.tensor(y_val, dtype=torch.float16)

    # Load dataset
    print("Load a batch dataset")
    train_iter = load_array((X_train, y_train), batch_size, is_train=True)
    val_iter = load_array((X_val, y_val), batch_size, is_train=False)
    print("===================================================\n")

    val_acc = train_ch6(VNet, train_iter, val_iter, num_epochs, lr, d2l.try_gpu(), is_real_train=False)
    val_acc_list.append(val_acc)
    val_acc_sum += val_acc

# Calculate the average accuracy on the validation dataset
val_acc_avg = val_acc_sum / 5

# Print the accuracy and average accuracy of each fold
for i, val_acc1 in enumerate(val_acc_list):
    print(f"The accuracy of Fold {i+1}: {val_acc1}")
print(f"\nThe average accuracy on the verification set after K-fold cross-validation: {round(val_acc_avg, 4)}\n")


# formal training begins without dividing the training set and the test set
print("Official training begins!")
print("Read Excel data and convert it to a matrix.")
features1984_1999, policy_labels1984_1999, value_labels1984_1999 = generate_dataset_allexcel3(
    r'C:\Users\Longwen-Liu\Desktop\洪水预处理\1984~1999年洪水数据预处理.xlsx', labels_preprocess=True)  # Change to your file path
features2000_2023, policy_labels2000_2023, value_labels2000_2023 = generate_dataset_allexcel3(
    r'C:\Users\Longwen-Liu\Desktop\洪水预处理\2000~2023年洪水数据预处理.xlsx', labels_preprocess=True)
features_05_1984, policy_labels_05_1984, value_labels_05_1984 = generate_dataset_allexcel3(
    r'C:\Users\Longwen-Liu\Desktop\洪水预处理\(0.5)1984~1999年洪水数据预处理.xlsx',
    labels_preprocess=True)
features_05_2000, policy_labels_05_2000, value_labels_05_2000 = generate_dataset_allexcel3(
    r'C:\Users\Longwen-Liu\Desktop\洪水预处理\(0.5)2000~2023年洪水数据预处理.xlsx',
    labels_preprocess=True)
features_08_1984, policy_labels_08_1984, value_labels_08_1984 = generate_dataset_allexcel3(
    r'C:\Users\Longwen-Liu\Desktop\洪水预处理\(0.8)1984~1999年洪水数据预处理.xlsx',
    labels_preprocess=True)
features_08_2000, policy_labels_08_2000, value_labels_08_2000 = generate_dataset_allexcel3(
    r'C:\Users\Longwen-Liu\Desktop\洪水预处理\(0.8)2000~2023年洪水数据预处理.xlsx',
    labels_preprocess=True)
features_12_1984, policy_labels_12_1984, value_labels_12_1984 = generate_dataset_allexcel3(
    r'C:\Users\Longwen-Liu\Desktop\洪水预处理\(1.2)1984~1999年洪水数据预处理.xlsx',
    labels_preprocess=True)
features_12_2000, policy_labels_12_2000, value_labels_12_2000 = generate_dataset_allexcel3(
    r'C:\Users\Longwen-Liu\Desktop\洪水预处理\(1.2)2000~2023年洪水数据预处理.xlsx',
    labels_preprocess=True)
features_15_1984, policy_labels_15_1984, value_labels_15_1984 = generate_dataset_allexcel3(
    r'C:\Users\Longwen-Liu\Desktop\洪水预处理\(1.5)1984~1999年洪水数据预处理.xlsx',
    labels_preprocess=True)
features_15_2000, policy_labels_15_2000, value_labels_15_2000 = generate_dataset_allexcel3(
    r'C:\Users\Longwen-Liu\Desktop\洪水预处理\(1.5)2000~2023年洪水数据预处理.xlsx',
    labels_preprocess=True)  # Change to your file path

# Connected data matrix
features_1 = np.concatenate(
    (features1984_1999, features2000_2023, features_05_1984, features_05_2000, features_08_1984), axis=0)
value_labels_1 = np.concatenate(
    (value_labels1984_1999, value_labels2000_2023, value_labels_05_1984, value_labels_05_2000, value_labels_08_1984),
    axis=0)

# Connected data matrix
features_2 = np.concatenate((features_08_2000, features_12_1984, features_12_2000, features_15_1984, features_15_2000),
                            axis=0)
value_labels_2 = np.concatenate((value_labels_08_2000, value_labels_12_1984, value_labels_12_2000, value_labels_15_1984,
                                 value_labels_15_2000), axis=0)

# features_1 = features1984_1999
# value_labels_1 = value_labels1984_1999
# features_2 = features2000_2023
# value_labels_2 = value_labels2000_2023

# assembly dataset
features_train = np.concatenate((features_1, features_2), axis=0)
value_labels_train = np.concatenate((value_labels_1, value_labels_2), axis=0)
features_val = np.array([])
value_labels_val = np.array([])

# Convert a matrix to a tensor
# train dataset
features_train = torch.tensor(features_train, dtype=torch.float16)
value_labels_train = torch.tensor(value_labels_train, dtype=torch.float16)
# validation dataset
features_val = torch.tensor(features_val, dtype=torch.float16)
value_labels_val = torch.tensor(value_labels_val, dtype=torch.float16)

# Load dataset
print("Load a batch dataset")
train_iter = load_array((features_train, value_labels_train), batch_size, is_train=True)
val_iter = load_array((features_val, value_labels_val), batch_size, is_train=False)
print("===================================================\n")

val_acc = train_ch6(VNet, train_iter, val_iter, num_epochs, lr, d2l.try_gpu(), is_real_train=True)
