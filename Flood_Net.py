# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
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
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PNet(nn.Module):
    """policy-value network module
       策略——价值网络的模型
    """

    def __init__(self, in_width, in_height, out_width, out_height):
        super(PNet, self).__init__()

        self.in_width = in_width
        self.in_height = in_height
        self.out_width = out_width
        self.out_height = out_height

        # 公共网络层
        self.conv1 = nn.Conv2d(24, 96, kernel_size=11, stride=2, padding=5)  # 第一层卷积层
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)  # 第二层卷积层
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        # self.conv3 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)  # 第三层卷积层
        # self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.flatten1 = nn.Flatten()

        # 策略网络特有层
        self.policy_fc1 = nn.Linear(4800, 1024)  # 第六层全连接层
        self.policy_dro1 = nn.Dropout(p=0.3)
        # self.policy_fc2 = nn.Linear(2048, 1024)  # 第六层全连接层
        # self.policy_dro2 = nn.Dropout(p=0.3)
        # self.policy_fc3 = nn.Linear(1024, 512)  # 第六层全连接层
        # self.policy_dro3 = nn.Dropout(p=0.3)
        self.policy_fc4 = nn.Linear(1024, self.out_width * self.out_height)  # 第八层全连接层

    def forward(self, state_input):
        # 公共网络层
        x = F.relu(self.conv1(state_input.float()))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        # x = F.relu(self.conv3(x))
        # x = self.maxpool3(x)
        x = self.flatten1(x)

        # 策略网络层
        # x = x.view(-1, 6400)
        x_policy = F.relu(self.policy_fc1(x))
        x_policy = self.policy_dro1(x_policy)
        # x_policy = F.relu(self.policy_fc2(x_policy))
        # x_policy = self.policy_dro2(x_policy)
        # x_policy = F.relu(self.policy_fc3(x_policy))
        # x_policy = self.policy_dro3(x_policy)
        x_policy = F.log_softmax(self.policy_fc4(x_policy), dim=1)

        # 返回输出
        return x_policy


class VNet(nn.Module):
    """policy-value network module
       策略——价值网络的模型
    """

    def __init__(self, in_width, in_height, out_width, out_height):
        super(VNet, self).__init__()

        self.in_width = in_width
        self.in_height = in_height
        self.out_width = out_width
        self.out_height = out_height

        # 公共网络层
        self.conv1 = nn.Conv2d(24, 96, kernel_size=11, stride=2, padding=5)  # 第一层卷积层
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)  # 第二层卷积层
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        # self.conv3 = nn.Conv2d(192, 192, kernel_size=3, padding=1)  # 第三层卷积层
        # self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.flatten1 = nn.Flatten()

        # 价值网络特有层
        self.value_fc1 = nn.Linear(4800, 1024)  # 第十四层全连接层
        self.value_dro1 = nn.Dropout(p=0.3)
        self.value_fc2 = nn.Linear(1024, 256)  # 第十五层全连接层
        self.value_dro2 = nn.Dropout(p=0.3)
        self.value_fc3 = nn.Linear(256, 64)  # 第十五层全连接层
        self.value_dro3 = nn.Dropout(p=0.3)
        self.value_fc4 = nn.Linear(64, 1)  # 第十八层全连接层

    def forward(self, state_input):
        # 公共网络层
        x = F.relu(self.conv1(state_input.float()))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        # x = F.relu(self.conv3(x))
        # x = self.maxpool3(x)
        x = self.flatten1(x)

        # 价值网络层
        # x = x.view(-1, 6400)
        x_value = F.relu(self.value_fc1(x))
        x_value = self.value_dro1(x_value)
        x_value = F.relu(self.value_fc2(x_value))
        x_value = self.value_dro2(x_value)
        x_value = F.relu(self.value_fc3(x_value))
        x_value = self.value_dro3(x_value)
        x_value = torch.tanh(self.value_fc4(x_value))

        # 返回输出
        return x_value


# 创建模型实例
# model = PVNet(out_width=20, out_height=20)
#
# X = torch.rand(size=(10, 24, 60, 60), dtype=torch.float32)
# # for name, layer in model.named_children():
# #     X = layer(X)
# #     print(name, 'output shape:', X.shape)
# policy, value = model(X)
# print(policy.shape)
# print(value.shape)


class PolicyValueNet(object):
    """policy-value network """

    def __init__(self, layer, in_width, in_height, out_width, out_height,
                 model_file1=None, model_file2=None, use_gpu=True):
        self.use_gpu = use_gpu  # 是否使用gpu
        self.layer = layer
        self.in_width = in_width
        self.in_height = in_height
        self.out_width = out_width
        self.out_height = out_height
        self.l2_const = 1e-4  # 罚分系数
        # 策略——价值网络模块
        if self.use_gpu:
            self.policy_net = PNet(in_width, in_height, out_width, out_height).cuda()
        else:
            self.policy_net = PNet(in_width, in_height, out_width, out_height)
        self.optimizer = optim.Adam(self.policy_net.parameters(), weight_decay=self.l2_const)

        if model_file1:
            net_params = torch.load(model_file1)
            self.policy_net.load_state_dict(net_params)

        # 价值网络模块
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
        输入:一批状态
        输出:一批动作概率和状态值
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
    # ***重要函数***
    def policy_value_fn(self, flood_board_copy, features_value_copy, is_outflow=True):
        """
        输入:flood_board
        输出:每个可用动作的(动作，概率)元组列表和flood_board状态的分数
        """
        # 获取可行的出库流量值
        if is_outflow:
            legal_positions = flood_board_copy.availables_outflow
            # print(f"availables_Outflow={legal_positions}")
        else:
            legal_positions = flood_board_copy.availables_inflow
            # print(f"availables_inflow={legal_positions}")
        # 将特征值填充为输入矩阵
        current_state = np.ascontiguousarray(flood_board_copy.current_state_auto(features_value_copy).reshape(
            -1, self.layer, self.in_width, self.in_height))
        # 是否使用GPU，分别计算了每个可能点的概率和奖励值
        if self.use_gpu:
            # print(torch.from_numpy(current_state).cuda().float().shape)
            log_act_probs = self.policy_net(torch.from_numpy(current_state).cuda().float())
            value = self.value_net(torch.from_numpy(current_state).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs = self.policy_net(torch.from_numpy(current_state).float())
            value = self.value_net(torch.from_numpy(current_state).cuda().float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        # 打包数据并返回
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]  # value只有一个值
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """执行一个训练步骤"""
        # 包装变量
        if self.use_gpu:
            # state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            # mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            # winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())

            state_batch = torch.FloatTensor(np.array(state_batch)).cuda()
            mcts_probs = torch.FloatTensor(np.array(mcts_probs)).cuda()
            winner_batch = torch.FloatTensor(np.array(winner_batch)).cuda()
        else:
            # state_batch = Variable(torch.FloatTensor(state_batch))
            # mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            # winner_batch = Variable(torch.FloatTensor(winner_batch))

            state_batch = torch.FloatTensor(np.array(state_batch))
            mcts_probs = torch.FloatTensor(np.array(mcts_probs))
            winner_batch = torch.FloatTensor(np.array(winner_batch))

        # 将参数梯度归零
        self.optimizer.zero_grad()
        # 设置学习率
        set_learning_rate(self.optimizer, lr)

        # print(state_batch.shape)

        # 前向传播
        log_act_probs, value = self.policy_net(state_batch)
        # print(log_act_probs.shape)
        # print(value)
        # print(value.shape)
        # print(winner_batch)
        # print(winner_batch.shape)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # 注意:L2惩罚包含在优化器中
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        # 反向传播和优化
        loss.backward()
        self.optimizer.step()
        # 计算策略熵，仅用于监控
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )
        # return loss.data[0], entropy.data[0]
        # for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
