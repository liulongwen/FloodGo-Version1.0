# -*- coding: utf-8 -*-
"""
human VS AI models
@author: Longwen Liu
"""

from __future__ import print_function
import numpy as np
from Flood_Rule import FloodBoard
from Flood_Players import PlayerInflow, PlayerOutflow, MCTSPlayerPure
from Flood_Net import PolicyValueNet  # Pytorch
# from mcts_pure import MCTSPlayerPure as MCTS_Pure
from calculate_tool.features import read_features_value
import random
from calculate_tool.data_preprocess1 import timefn


class FloodGame(object):
    """
    Flood control operation
    """

    def __init__(self, flood_board):
        self.flood_board = flood_board

    @timefn
    # Game between the predicted inbound flow and the predicted outbound flow program
    def start_play(self, player1, player2, features_value_init, start_player=0, is_shown=True):
        # Initialize checkerboard
        self.flood_board.init_flood_board(features_value_init, start_player)
        print("the initial feasible solution：", self.flood_board.availables_outflow)

        # Save the player as 1 and 2 for later reference. In this procedure take 1:inflow, 2:outflow.
        p1, p2 = self.flood_board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        features_value = features_value_init
        count = 0
        time = 0
        end = False
        while True:
            # Gets who the current player is, out or in
            current_player = self.flood_board.get_current_player()
            if count == 0:
                print("time={}".format(time))

            # The current player takes action
            # inflow
            if current_player == 1:
                print("current_player: {}".format(current_player))
                inflow = player1.get_flow(features_value, time)
                features_value = self.flood_board.do_inflow(features_value, time, inflow, is_shown=False)
                inflow_sort = self.flood_board.flow_to_sort(inflow)
                print("inflow label: ", inflow_sort)
                print("inflow: ", inflow)
                print("Feasible solution after removing water level limit：", self.flood_board.availables_outflow)
            # 出库流量
            if current_player == 2:
                print("current_player: {}".format(current_player))
                outflow_sort = player2.get_outflow_sort(features_value, time, count)
                outflow = self.flood_board.sort_to_flow(outflow_sort)
                features_value = self.flood_board.do_outflow(features_value, time, outflow, is_shown=False)
                print("outflow label: ", outflow_sort)
                print("outflow: ", outflow)
                print("water level above the dam: ", features_value[time, 2], "\n")

                # Check whether the scheduling process is complete
                end = self.flood_board.flood_end_time(features_value, time, is_self=False)

            # time +1
            count += 1
            if count % 2 == 0:
                time = time + 1
                print("time={}".format(time))

            # output result
            if end:
                # Calculating flood control effect
                flood_value, index1, index2, index3, index4, index5, index6, index7, index8, index9 = self.flood_board.calculate_flood_evaluate(
                    features_value, return_index=True)
                with open(r'C:\APP\Python\FloodGo-V1.0\flood_MCTS\flood_result\result_5%.txt', 'w') as file:
                    file.truncate(0)
                with open(r'C:\APP\Python\FloodGo-V1.0\flood_MCTS\flood_result\result_5%.txt', 'w') as file:
                    for i in range(len(features_value[:, 0])):
                        result_txt = f'{features_value[i, 0]}, {features_value[i, 1]}, {features_value[i, 2]}, {features_value[i, 3]}\n'
                        file.write(result_txt)
                # Print display result
                if is_shown:
                    print("inflow：", features_value[:, 0])
                    print("outflow：", features_value[:, 1])
                    print("water level above the dam：", features_value[:, 2])
                    print("water level below the dam：", features_value[:, 3], "\n")
                    print(f"maximum combined flow of the control point：{index1:.4f}, {index4:.0f}, {index5:.0f}")
                    print(f"use flood storage：{index2:.4f}, {index6:.0f}, {index7:.0f}")
                    print(f"Difference between final water level and target water level：{index3:.4f}, {index8:.2f}, {index9:.2f}")
                    print(f"Comprehensive flood control effect score：{flood_value:.4f}")
                    break


def run():
    # Sets the initialized feature matrix data
    layer = 24
    in_width, in_height = 60, 60
    out_width, out_height = 20, 10
    features_num = 16

    # value network weight parameter
    model_file1 = r'C:\APP\Python\FloodGo-V1.0\flood_MCTS\model_weight\using_best_policy.model'
    # value network weight parameter
    model_file2 = r'C:\APP\Python\FloodGo-V1.0\flood_MCTS\model_weight\using_best_value.model'

    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    np.set_printoptions(suppress=True)
    try:
        # Create two instance objects flood_board and flood_game
        flood_board = FloodBoard(layer=layer, in_width=in_width, in_height=in_height,
                                 out_width=out_width, out_height=out_height, features_num=features_num)
        flood_game = FloodGame(flood_board)

        # Obtain the initial data of reservoir dispatching
        inflow_all, interval_flow, z_up, z_down, flood_constant_feature = \
            read_features_value(r'C:\APP\Python\FloodGo-V1.0\flood_MCTS\flood_data\features_value_5%.txt')

        # Use an array to hold the eigenvalues
        features_value_init = flood_board.initialize_feature_value(inflow_all, interval_flow, z_up,
                                                                   z_down, flood_constant_feature)

        # Create an inbound inflow player
        player_inflow = PlayerInflow()

        # Create an outbound outflow player based on Monte Carlo tree search and convolutional neural networks
        best_policy = PolicyValueNet(layer, in_width, in_height, out_width, out_height, model_file1=model_file1,
                                     model_file2=model_file2, use_gpu=True)
        player_outflow = PlayerOutflow(best_policy.policy_value_fn, flood_board, c_puct=2, n_playout=1000)

        # Uncomment the following line to use pure MCTSPure(which is much weaker even with larger n_playout)
        # player_outflow = MCTSPlayerPure(flood_board, c_puct=2, n_playout=1000)

        # Start the game
        flood_game.start_play(player_inflow, player_outflow, features_value_init)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
