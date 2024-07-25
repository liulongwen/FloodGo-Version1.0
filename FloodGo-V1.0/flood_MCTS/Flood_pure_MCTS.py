# -*- coding: utf-8 -*-
"""
A pure implementation of Monte Carlo tree Search (MCTSPure)
@author: Longwen Liu
"""

import numpy as np
import copy
from operator import itemgetter
from calculate_tool.data_preprocess1 import timefn


def rollout_policy_fn(flood_board, is_outflow=True):
    """
    A rough, fast version of policy_fn used in the rollout phase.
    """
    if is_outflow:
        action_probs = np.random.rand(len(flood_board.availables_outflow))
    else:
        action_probs = np.random.rand(len(flood_board.availables_inflow))
    return zip(flood_board.availables_outflow, action_probs)


def policy_value_fn_pure(flood_board_copy, features_value_copy, is_outflow=True):
    """
    A function that accepts a state and outputs a list of tuples (actions, probabilities) and a state score
    """
    # For pure MCTSPure, the uniform probability and 0 points are returned
    action_probs = np.ones(len(flood_board_copy.availables_outflow)) / len(flood_board_copy.availables_outflow)
    return zip(flood_board_copy.availables_outflow, action_probs), 0


class TreeNodePure(object):
    """
    A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and prior fraction u adjusted for access counts
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNodePure
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, flow_priors):
        """
        Extend the tree by creating new child nodes.
        Action_priors: A list of operation tuples and their prior probabilities based on the policy function.
        """
        for action, prob in flow_priors:
            if action not in self._children:
                self._children[action] = TreeNodePure(self, prob)

    def select(self, c_puct):
        """
        Choose the action in the child that yields the maximum action value Q plus the reward u(P).
        Returns a tuple of :(flow, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        Update node values from leaf calculations.
        Leaf_value: The value that the subtree evaluates from the current player's point of view.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """
        Similar to the call to update(), but applied recursively to all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        Calculates and returns the value of the node. It is a combination of the leaf evaluation value Q and
        the prior value u that this node adjusts based on its number of visits.
        c_puct: A number in (0, inf), the relative influence of the control value Q and the prior probability P
        on the score of this node.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """
        Check the leaf nodes (that is, the following nodes are not extended).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTSPure(object):
    """
    Implementation of pure Monte Carlo tree search.
    """

    def __init__(self, policy_value_fn, flood_board, c_puct=5, n_playout=10000):
        """
        Policy_value_fn: A function that accepts the flood_board state and outputs a list of (action, probability) tuples,
        And the current player's [-1,1] score (the expected final game score from the current player's point of view).
        C_puct: A number in (0, inf) that controls the speed at which exploration converges to the maximum policy.
        The higher the value, the greater the reliance on the prior.
        """
        self._root = TreeNodePure(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.flood_board = flood_board

    def _playout(self, flood_board_copy, features_value_copy, time, count, period=100, is_self=False):
        """
        A play is performed from the root node to the leaf node, taking a value on the leaf node and
        propagating it back through its parent.
        The state is modified locally, so a copy must be provided.
        """
        # The first node is named the root node
        node = self._root
        global leaf_value
        flow_probs = np.zeros(flood_board_copy.out_width * flood_board_copy.out_height)
        time_start = time
        count_start = count
        while 1:
            if node.is_leaf() or time_start == len(features_value_copy[:, 0]):
                break

            # Greedily choose the next step.
            # inflow and outflow
            flow_sort, node = node.select(self._c_puct)
            flow = flood_board_copy.sort_to_flow(flow_sort)

            # update features_value
            if count_start % 2 == 0:
                features_value_copy = flood_board_copy.do_inflow_playout(features_value_copy, time_start, flow)
            if count_start % 2 == 1:
                features_value_copy = flood_board_copy.do_outflow_playout(features_value_copy, time_start, flow)

            # update time
            count_start = count_start + 1
            if count_start % 2 == 0:
                time_start = time_start + 1

        """
        A leaf is evaluated using a network that outputs a list of the (flow, probability) tuple p and the current
        player's fraction leaf_value in [-1,1].
        Neural networks are used to find the probability of feasible solutions.
        """
        if count_start % 2 == 0:
            flow_probs, _ = self._policy(flood_board_copy, features_value_copy, is_outflow=False)
        if count_start % 2 == 1:
            flow_probs, _ = self._policy(flood_board_copy, features_value_copy, is_outflow=True)

        # Check the flood control operation is complete
        end = flood_board_copy.flood_end_count(features_value_copy, count_start, period, is_self)
        if not end:
            node.expand(flow_probs)

        # The flood control effect of leaf node is calculated by random rollout
        leaf_value = self._evaluate_rollout(flood_board_copy, features_value_copy, time, count, is_self)
        # leaf_value = leaf_value.cpu().numpy()  # Comments section plus, 5 times faster

        # Update the values and access counts of the nodes in the traverse.
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, flood_board_copy, features_value_copy, time, count, period=100, is_self=False, limit=1000):
        """
        Use the rollout strategy to play until the end of the game, if the current player wins, return +1,
        Returns -1 if the opponent wins, 0 if it is a draw.
        """
        global action_probs
        time_start = time
        count_start = count
        for i in range(limit):
            # Find the best action value (flow)
            if count_start % 2 == 0:
                action_probs = rollout_policy_fn(flood_board_copy, is_outflow=False)
            if count_start % 2 == 1:
                action_probs = rollout_policy_fn(flood_board_copy, is_outflow=True)
            max_action_sort = max(action_probs, key=itemgetter(1))[0]
            max_action_flow = flood_board_copy.sort_to_flow(max_action_sort)

            if count_start % 2 == 0:
                features_value_copy = flood_board_copy.do_inflow_playout(features_value_copy, time, max_action_flow)
            if count_start % 2 == 1:
                features_value_copy = flood_board_copy.do_inflow_playout(features_value_copy, time, max_action_flow)
            # Check the flood control operation is complete
            end = flood_board_copy.flood_end_count(features_value_copy, count_start, period, is_self)
            if end:
                break
            # update time
            count_start = count_start + 1
            if count_start % 2 == 0:
                time_start = time_start + 1
        else:
            print("Warning: Flood control time limit reached!")

        # Calculate the value of the flood control effect
        leaf_value = flood_board_copy.calculate_flood_evaluate(features_value_copy)
        # Get current player
        current_player = flood_board_copy.get_current_player()
        if current_player == 1:
            leaf_value = -1.0 * leaf_value
        if current_player == 2:
            leaf_value = 1.0 * leaf_value
        return leaf_value

    @timefn
    def get_inflow(self, features_value, time, count, period=100, is_self=False):
        """
        Run all plays in order and return the most accessed actions.
        Status: Current game status
        Return: Selected action
        """
        for n in range(self._n_playout):
            flood_board_copy = copy.deepcopy(self.flood_board)
            features_value_copy = copy.deepcopy(features_value)
            # print("第{}次pure _n_playout".format(n + 1))
            self._playout(flood_board_copy, features_value_copy, time, count,  period, is_self)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """
        Go one step further and keep all the information we know about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNodePure(None, 1.0)

    def __str__(self):
        return "MCTSPure"

