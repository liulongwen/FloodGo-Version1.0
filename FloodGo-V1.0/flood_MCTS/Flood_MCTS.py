# -*- coding: utf-8 -*-
"""
@author: Longwen Liu
"""

import numpy as np
import copy
from calculate_tool.data_preprocess1 import timefn
from calculate_tool.labels_restore import labels_restore4_predict
from d2l import torch as d2l


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """
    A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and prior fraction u adjusted for access counts
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, flow_priors):
        """
        Extend the tree by creating new child nodes.
        flow_priors: A list of action tuples and their prior probabilities according to the policy function.
        """
        for flow, prob in flow_priors:
            if flow not in self._children:
                self._children[flow] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        Choose the action in the child that yields the maximum action value Q plus the reward u(P).
        Returns a tuple of :(flow, next_node)
        """
        # lambda is a simple anonymous function with lambda arguments: expressions.
        return max(self._children.items(), key=lambda flow_node: flow_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        Update node values from leaf calculations.
        Leaf_value: The value that the subtree evaluates from the current player's point of view.
        """
        # Statistical number of visits
        self._n_visits += 1
        # Update Q, the running average of all access values.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """
        Similar to the call to update(), but applied recursively to all ancestors.
        """
        # if it is not the root node, the parent of the node should be updated first
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        Calculates and returns the value of the node. It is a combination of the leaf evaluation
        value Q and the prior value u that this node adjusts based on its number of visits.
        c_puct: A number in (0, inf), the relative influence of the control value Q and the prior
        probability P on the score of this node.
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """
        Check the leaf nodes (that is, the following nodes are not extended).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """
    Monte Carlo tree search implementation.
    """

    def __init__(self, policy_value_fn, flood_board, c_puct=5, n_playout=10000):
        """
        Policy_value_fn: A function that accepts the flood_board state and outputs a list of tuples (action, probability),
         along with the current player's [-1,1] score (i.e. the expected value of the final game score from the current
         player's point of view).
         C_puct: A number in (0, inf) that controls the speed at which exploration converges to the maximum policy.
         The higher the value, the greater the reliance on the prior.
        """

        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.flood_board = flood_board

    # @timefn
    def _playout(self, flood_board_copy, features_value_copy, time, count, period=100, is_self=False):
        """
        Run the play once from the root node to the leaf node, taking a value on the leaf node and propagating
        it back through the parent node.
        The state is modified locally, so a copy must be provided.
        """
        # The first node is named the root node
        node = self._root
        global leaf_value
        flow_probs = np.zeros(flood_board_copy.out_width * flood_board_copy.out_height)
        time_start = time
        count_start = count
        while 1:
            # The loop ends when it is a leaf node
            if node.is_leaf():
                break

            # inflow and outflow
            flow_sort, node = node.select(self._c_puct)
            flow = flood_board_copy.sort_to_flow(flow_sort)

            # upgrade features_value
            if count_start % 2 == 0:
                features_value_copy = flood_board_copy.do_inflow_playout(features_value_copy, time_start, flow)
            if count_start % 2 == 1:
                features_value_copy = flood_board_copy.do_outflow_playout(features_value_copy, time_start, flow)

            # upgrade time
            count_start = count_start + 1
            if count_start % 2 == 0:
                time_start = time_start + 1

        """
        A leaf is evaluated using a network that outputs a list of the (flow, probability) tuple p and the 
        current player's fraction leaf_value in [-1,1].
        Neural networks are used to find the probability of feasible solutions.
        This is to find probs and leaf_value for the next step.
        """
        if count_start % 2 == 0:
            flow_probs, leaf_value = self._policy(flood_board_copy, features_value_copy, is_outflow=False)

        if count_start % 2 == 1:
            flow_probs, leaf_value = self._policy(flood_board_copy, features_value_copy, is_outflow=True)

        # Check the flood control operation is complete
        leaf_value = leaf_value.cpu().numpy()  # Comments section plus, 5 times faster
        end = flood_board_copy.flood_end_count(features_value_copy, count_start, period, is_self)
        if not end:
            node.expand(flow_probs)
        else:
            leaf_value = flood_board_copy.calculate_flood_evaluate(features_value_copy)

        # The leaf_value value is specified according to the result
        if count_start % 2 == 0:
            leaf_value = -1.0 * leaf_value
        if count_start % 2 == 1:
            leaf_value = 1.0 * leaf_value

        # Update the values and access counts of the nodes in the traverse.
        node.update_recursive(-leaf_value)

    # @timefn
    def get_outflow_probs(self, features_value, time, count, period=100, is_self=False, temp=1e-3):
        """
        Run all plays in order and return the available actions and their corresponding probabilities.
        state: indicates the current game status
        The temperature parameter in temp:(0,1) controls the degree of detection
        """
        # start simulation scheduling
        for n in range(self._n_playout):  # _n_playout=10000
            flood_board_copy = copy.deepcopy(self.flood_board)  # make a copy
            features_value_copy = copy.deepcopy(features_value)  # make a copy
            if (n+1) % 500 == 0:
                print("{}th simulated scheduling".format(n + 1))
            self._playout(flood_board_copy, features_value_copy, time, count, period, is_self)  # simulation

        # the movement probability is calculated based on the number of accesses to the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_outflow(self, last_outflow):
        """
        Go one step further and keep all the information we know about the subtree.
        """
        if last_outflow in self._root._children:
            self._root = self._root._children[last_outflow]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTSPure"

    def max_probs(self, act_probs_zip):
        max_prob = float('-inf')
        max_positions = 0

        for position, probability in act_probs_zip:
            if probability > max_prob:
                max_prob = probability
                max_positions = position
        return max_positions, max_prob
