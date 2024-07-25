# -*- coding: utf-8 -*-
"""
@author: Longwen Liu
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from Flood_Rule import FloodBoard
from Flood_Game import FloodGame
from Flood_Players import MCTSPlayerPure as MCTS_Pure
from Flood_Players import PlayerOutflow
from Flood_Net import PolicyValueNet  # Pytorch

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.set_printoptions(suppress=True)


class TrainProcess(object):
    def __init__(self, init_model=None):
        # params of the flood_board and the game
        self.layer = 24
        self.in_width = 80
        self.in_height = 80
        self.out_width = 20
        self.out_height = 20
        self.features_num = 15
        # self.n_in_row = 5
        self.flood_board = FloodBoard(layer=self.layer, in_width=self.in_width, in_height=self.in_height,
                                      out_width=self.out_width, out_height=self.out_height,
                                      features_num=self.features_num)
        self.game = FloodGame(self.flood_board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # Adaptive adjustment of learning rate based on KL
        self.temp = 1.0  # Temperature parameter
        self.n_playout = 400  # The number of simulations per move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # Training for small batch sizes
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 10  # The number of iterations per update of parameter weights
        self.kl_targ = 0.02
        self.check_freq = 2  # The number of intervals between saving model parameters
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # In the number of simulations of pure McTs_pure, it is used to evaluate the training strategy of the opponent
        self.pure_mcts_playout_num = 1000
        self.period = 24  # The duration of the game
        self.n_games = 5  # The number of games played against pure Monte Carlo tree search
        if init_model:
            # Start training with the initial policy-value network
            self.policy_value_net = PolicyValueNet(self.layer, self.in_width, self.in_height, self.out_width, self.out_height,
                                                   model_file=init_model)
        else:
            # Start training with a new policy value network
            self.policy_value_net = PolicyValueNet(self.layer, self.in_width, self.in_height, self.out_width, self.out_height)
        # Create a player instance using MCTS
        self.mcts_player = PlayerOutflow(self.policy_value_net.policy_value_fn, self.flood_board, c_puct=self.c_puct,
                                         n_playout=self.n_playout, is_selfplay=1)

    # Collect self-game data
    def collect_selfplay_data(self, n_games=1, period=100):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, period=period, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)

    # Update Strategy - Value Network
    def policy_value_update(self):
        """update the policy-value net"""
        print("数据量data_buffer为: {}".format(len(self.data_buffer)))
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            # Update model parameters
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch,
                                                             winner_batch, self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            # KL divergence (kl), which is an early stop strategy, stops training when the KL divergence deviates
            # significantly from the target value to avoid problems caused by continuing training.
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # dynamically adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # to interpret the variance, the closer to 1 the better, and the closer to 0 the worse
        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        # Print various indicator values
        # The smaller the loss, the better, the closer the entropy is to 0 the more stable the model results,
        # the closer the explained_var explains the variance, the better the closer the explained_var the worse
        print(("kl:{:.5f}, "
               "lr_multiplier:{:.3f}, "
               "loss:{}, "
               "entropy:{}, "
               "explained_var_old:{:.3f}, "
               "explained_var_new:{:.3f}"
               ).format(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))

        # Returns loss and entropy
        return loss, entropy

    # Evaluation Strategy - Effectiveness of value network
    def policy_value_evaluate(self, n_games=10, period=100):
        """
        Evaluate training strategies by competing against pure MCTS players.
        Note: This is to monitor the progress of the training only.
        """
        # Combine policy-value network and Monte Carlo tree search for players
        current_mcts_player = PlayerOutflow(self.policy_value_net.policy_value_fn, self.flood_board,
                                            c_puct=self.c_puct, n_playout=self.n_playout)
        # Pure Monte Carlo tree search for players
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(float)
        flood_value_list = []
        features_value_list = []
        for i in range(n_games):
            flood_constant_feature = []
            inflow = random.randint(100, 1000)
            z_up = round(random.uniform(261.0, 276.0), 2)
            z_down = z_up - round(random.uniform(50.0, 65.0), 2)
            flood_constant_feature.append(random.choice([0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]))
            flood_constant_feature.append(random.randint(-1, 1))
            flood_constant_feature.append(random.randint(-1, 1))
            flood_constant_feature.append(round(random.uniform(268.0, 273.0), 2))
            flood_constant_feature.append(round(280.0, 2))
            features_value = self.flood_board.initialize_feature_value(inflow, z_up, z_down, flood_constant_feature,
                                                                       is_inflow_all=False, period=period)
            features_value_list.append(features_value)

        # Start the game
        for i in range(n_games):
            flood_value, winner = self.game.start_play_test(pure_mcts_player, current_mcts_player, features_value_list[i],
                                                            period=period)
            print(flood_value_list[i])
            # flood_value = self.flood_board.calculate_flood_evaluate(features_value_list[i])
            flood_value_list.append(flood_value)
            win_cnt[winner] += flood_value
            print("Game{} is over".format(i + 1))
        # Calculate the flood control effect of Player 2 (outflow)
        win_ratio = 1.0 * win_cnt[2] / n_games
        # Print competition results
        print("The number of searches in a pure Monte Carlo tree: {}, "
              "10 simulated flood control effects: {} \n "
              "10 simulated flood control effects: {}".format
              (self.pure_mcts_playout_num, win_ratio, flood_value_list))
        # Return flood control effect for Player 2 (outflow)
        return win_ratio

    # The main function of training
    def run(self):
        """
        run the training process
        """
        try:
            print("Start of model training:")
            for i in range(self.game_batch_num):  # Set number of iterations
                self.collect_selfplay_data(self.play_batch_size, period=self.period)  # Collect self-game data
                print("Number of offices for self-simulated scheduling i: {}, "
                      "The length of time for self-simulated scheduling T: {}".format
                      (i + 1, self.episode_len + 1))  # Print the number of plays and the length of the game
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_value_update()  # Update model parameters
                # Check the effect of the current model and save the model parameters
                if (i + 1) % self.check_freq == 0:  # Set the number of times a parameter is saved
                    print("\nSave model weight:")
                    print("The number of innings now scheduled by self-simulation i: {}".format(i + 1))
                    win_ratio = self.policy_value_evaluate(n_games=self.n_games, period=self.period)  # Calculate the win ratio of the current model
                    self.policy_value_net.save_model('./current_policy.model')  # Save the parameter file for the current model
                    if win_ratio > self.best_win_ratio:  # Compare the winning rate of the current model and the best model, and save the best one
                        print("A better strategy has emerged - the value network!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./using_best_policy.model')  # Save the parameter file of the optimal model
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:  # Improve the strength of pure Monte Carlo tree search players and further improve the training effect
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                    print('\n')
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_process = TrainProcess()
    training_process.run()

