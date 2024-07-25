# -*- coding: utf-8 -*-
"""
PlayerInflow VS AI models
@author: Longwen Liu
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from Flood_MCTS import MCTS
from Flood_pure_MCTS import MCTSPure, policy_value_fn_pure
from calculate_tool.data_preprocess1 import timefn


class PlayerInflow(object):
    """
    PlayerInflow player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_flow(self, features_value, time):
        inflow = features_value[time, 0]
        if inflow == np.NaN or inflow < 0:
            print("invalid flow!")
            inflow = -1
        return inflow

    def __str__(self):
        return "PlayerInflow {}".format(self.player)


class PlayerOutflow(object):
    """
    AI player based on MCTSPure
    """

    def __init__(self, policy_value_function, flood_board, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, flood_board, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.flood_board = flood_board

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_outflow(-1)

    @timefn
    def get_outflow_sort(self, features_value, time, count, period=100, is_self=False, temp=1e-3, return_prob=0):
        # Obtain a valid outflow value
        sensible_flows_sort = self.flood_board.availables_outflow
        # Initialize the output matrix
        outflow_probs = np.zeros(self.flood_board.out_width * self.flood_board.out_height)
        # Start Monte Carlo tree search
        if len(sensible_flows_sort) > 0:
            # Calculate action values and probabilities
            outflows_sort, probs = self.mcts.get_outflow_probs(features_value, time, count, period, is_self, temp)
            outflow_probs[list(outflows_sort)] = probs
            # Determine whether it is a game of self
            if self._is_selfplay:
                # Add Dirichlet noise for exploration (requires game training)
                outflow = np.random.choice(outflows_sort,
                                           p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                # Update the root node and reuse the search tree
                self.mcts.update_with_outflow(outflow)
            else:
                # For the default temp=1e-3, this is almost the same as choosing the move with the highest probability
                outflow = np.random.choice(outflows_sort, p=probs)
                # Reset root node
                self.mcts.update_with_outflow(-1)

            if return_prob:
                return outflow, outflow_probs
            else:
                return outflow
        else:
            print("Warning: There is no feasible solution for outflow!")

    def __str__(self):
        return "MCTSPure {}".format(self.player)


class MCTSPlayerPure(object):
    """
    AI player based on MCTSPure
    """

    def __init__(self, flood_board, c_puct=5, n_playout=2000):
        self.mcts_pure = MCTSPure(policy_value_fn_pure, flood_board, c_puct, n_playout)
        self.flood_board = flood_board

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts_pure.update_with_move(-1)

    def get_outflow_sort(self, features_value, time, count, period=100, is_self=True):
        # Obtain a valid outflow value
        sensible_flows_sort = self.flood_board.availables_outflow
        # Start Monte Carlo tree search
        if len(sensible_flows_sort) > 0:
            # Calculate inflow
            inflow_sort = self.mcts_pure.get_inflow(features_value, time, count, period, is_self)
            self.mcts_pure.update_with_move(-1)
            return inflow_sort
        else:
            print("WARNING: the flood is end")

    def __str__(self):
        return "MCTSPure {}".format(self.player)


