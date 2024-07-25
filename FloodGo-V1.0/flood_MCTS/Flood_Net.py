# -*- coding: utf-8 -*-
"""
@author: Longwen Liu
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.autograd import Variable
import numpy as np
# from Flood_Rule import FloodBoard
from calculate_tool.data_preprocess1 import timefn


def set_learning_rate(optimizer, lr):
    """
    Sets the learning rate to the given value
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PNet(nn.Module):
    """
    policy-value train_network module
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
        x_policy = F.log_softmax(self.policy_fc4(x_policy), dim=1)

        return x_policy


class VNet(nn.Module):
    """
    policy-value train_network module
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


class PolicyValueNet(object):
    """policy-value train_network """

    def __init__(self, layer, in_width, in_height, out_width, out_height,
                 model_file1=None, model_file2=None, use_gpu=True):
        self.use_gpu = use_gpu  # Whether to use gpu
        self.layer = layer
        self.in_width = in_width
        self.in_height = in_height
        self.out_width = out_width
        self.out_height = out_height
        self.l2_const = 1e-4
        # policy Network module
        if self.use_gpu:
            self.policy_net = PNet(in_width, in_height, out_width, out_height).cuda()
        else:
            self.policy_net = PNet(in_width, in_height, out_width, out_height)
        self.optimizer = optim.Adam(self.policy_net.parameters(), weight_decay=self.l2_const)

        if model_file1:
            net_params = torch.load(model_file1)
            self.policy_net.load_state_dict(net_params)

        # value Network module
        if self.use_gpu:
            self.value_net = VNet(in_width, in_height, out_width, out_height).cuda()
        else:
            self.value_net = VNet(in_width, in_height, out_width, out_height)
        self.optimizer = optim.Adam(self.value_net.parameters(), weight_decay=self.l2_const)

        if model_file2:
            net_params = torch.load(model_file2)
            self.value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        Input: batch status
        Output: A batch of action probabilities and status values
        """
        if self.use_gpu:
            # state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            state_batch = torch.FloatTensor(np.array(state_batch)).cuda()
            log_act_probs, value = self.policy_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            # state_batch = Variable(torch.FloatTensor(state_batch))
            state_batch = torch.FloatTensor(np.array(state_batch))
            log_act_probs, value = self.policy_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    # @timefn
    # *** Important functions ***
    def policy_value_fn(self, flood_board_copy, features_value_copy, is_outflow=True):
        """
        Input :flood_board
        Output: A (action, probability) tuple list for each available action and a score of the flood_board status
        """
        # Obtain a feasible outbound outflow value
        if is_outflow:
            legal_positions = flood_board_copy.availables_outflow
            # print(f"availables_Outflow={legal_positions}")
        else:
            legal_positions = flood_board_copy.availables_inflow
            # print(f"availables_inflow={legal_positions}")
        # Populate the eigenvalues as an input matrix
        current_state = np.ascontiguousarray(flood_board_copy.current_state_auto(features_value_copy).reshape(
            -1, self.layer, self.in_width, self.in_height))
        # The probability and reward value of each possible point are calculated separately, whether the GPU is used or not
        if self.use_gpu:
            # print(torch.from_numpy(current_state).cuda().float().shape)
            log_act_probs = self.policy_net(torch.from_numpy(current_state).cuda().float())
            value = self.value_net(torch.from_numpy(current_state).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs = self.policy_net(torch.from_numpy(current_state).float())
            value = self.value_net(torch.from_numpy(current_state).cuda().float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        # Package the data and return it
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """
        Perform a training step
        """
        # 包装变量
        if self.use_gpu:
            state_batch = torch.FloatTensor(np.array(state_batch)).cuda()
            mcts_probs = torch.FloatTensor(np.array(mcts_probs)).cuda()
            winner_batch = torch.FloatTensor(np.array(winner_batch)).cuda()
        else:
            state_batch = torch.FloatTensor(np.array(state_batch))
            mcts_probs = torch.FloatTensor(np.array(mcts_probs))
            winner_batch = torch.FloatTensor(np.array(winner_batch))

        # Zero the parameter gradient
        self.optimizer.zero_grad()
        # Set learning rate
        set_learning_rate(self.optimizer, lr)

        # Forward propagation
        log_act_probs, value = self.policy_net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        # Backpropagation and optimization
        loss.backward()
        self.optimizer.step()
        # Calculate policy entropy for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """
         save model params to file
         """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
