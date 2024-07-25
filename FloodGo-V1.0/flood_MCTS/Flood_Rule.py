# -*- coding: utf-8 -*-
"""
@author: Longwen Liu
"""

from __future__ import print_function

import copy

import numpy as np
from calculate_tool import data_preprocess3
from calculate_tool import data_preprocess1
from calculate_tool.Z_V import z_to_v, v_to_z
from calculate_tool.features import read_features_value
from calculate_tool.data_preprocess1 import timefn


# from Flood_Game import FloodGame


class FloodBoard(object):
    """flood_board for the game"""

    def __init__(self, **kwargs):
        self.layer = int(kwargs.get('layer', 24))
        self.in_width = int(kwargs.get('in_width', 60))
        self.in_height = int(kwargs.get('in_height', 60))
        self.out_width = int(kwargs.get('out_width', 20))
        self.out_height = int(kwargs.get('out_height', 10))
        self.features_num = int(kwargs.get('features_num', 16))
        self.states = {}
        self.players = [1, 2]  # player1 and player2

    def init_flood_board(self, features_value_init, start_player=0):
        self.current_player = self.players[start_player]  # start player
        # List the available actions in a list
        # inflow
        inflow_init = features_value_init[0, 0]
        inflow_init_sort = self.flow_to_sort(inflow_init)
        self.availables_inflow = [inflow_init_sort]

        # outflow
        inflow = features_value_init[:, 0]
        inflow_max = max(inflow)
        inflow_max_sort = self.flow_to_sort(inflow_max)
        # inflow_max_sort = int(inflow_max_sort*2/3)
        self.availables_outflow = list(range(0, inflow_max_sort + 1, 1))

        self.last_sort = -1

    # @timefn
    # Obtain feasible solution
    def get_available(self, flow, interval, section=0):
        available = []
        flow_sort = self.flow_to_sort(flow)
        flow_sort_left = flow_sort - interval
        flow_sort_right = flow_sort + interval
        if flow_sort - interval < 0:
            flow_sort_left = 0
        if flow_sort + interval > 400:
            flow_sort_right = 400
        # Determine the feasible solution interval
        if section == -1:
            available = list(range(flow_sort_left, flow_sort + 1))
        if section == 0:
            available = list(range(flow_sort_left, flow_sort_right + 1))
        if section == 1:
            available = list(range(flow_sort, flow_sort_right + 1))
        return available

    # @timefn
    # Add water limit (flood limit and flood high water)
    def get_available_consider_Z_limit(self, features_value, inflow, Z1, Z_limit=245.0, Z_high=275.0):
        inflow_all = features_value[:, 0]
        inflow_max = max(inflow_all)
        inflow_max_sort = self.flow_to_sort(inflow_max)
        # inflow_max_sort = int(inflow_max_sort * 2 / 3)
        availables_outflow = list(range(0, inflow_max_sort + 1, 1))
        availables_outflow_copy = copy.deepcopy(availables_outflow)
        for element in availables_outflow:
            outflow = self.sort_to_flow(element)
            V1 = z_to_v(Z1)
            V2 = V1 + (inflow - outflow) * 3600 / 10000
            Z2 = v_to_z(V2)
            # print(f'Z2={Z2}')
            if Z2 > Z_high or Z2 < Z_limit:
                availables_outflow_copy.remove(element)
        return availables_outflow_copy

    # @timefn
    # Added drain fluctuation limit
    def get_available_consider_outflow_trend_limit(self, outflow_trend, Tc):
        temp_sum = 0
        for temp in outflow_trend:
            temp_sum = temp_sum + temp
        if temp_sum == Tc or temp_sum == -Tc:
            section = 0
        else:
            section = outflow_trend[Tc - 1]
        return section

    def get_inflow_time(self, features_value, time):
        inflow = features_value[time, 0]
        if inflow == np.NaN or inflow < 0:
            print("invalid flow!")
            inflow = -1
        return inflow

    # @timefn
    # Calculate the discharge fluctuation trend value
    def calculate_outflow_trend_value(self, features_value, time, Tc=5):
        trend_list = [0] * Tc
        # print("time=", time)
        if time - (Tc - 1) <= 0:
            trend_list = [1] * Tc
        else:
            # print(1)
            data_outflow = features_value[time - Tc:time + 1, 1]
            # print(data_outflow)
            for i in range(Tc):
                outflow_variation = data_outflow[i + 1] - data_outflow[i]
                if -50 <= outflow_variation <= 50:
                    if i == 0:
                        value = 1
                    else:
                        value = trend_list[i - 1]
                elif outflow_variation > 50:
                    value = 1
                else:
                    value = -1
                trend_list[i] = value
        return trend_list

    def flow_to_sort(self, flow):
        """
        Divide 0~10000 into 20*20 categories
        eg: flow=1234 converts to 1234/50-1=48.36, rounded to class 48.
        flow=990 converts to 990/50-1=38.6, rounded to class 39.
        """
        temp = round(int(flow) / 50) - 1
        if temp == -1:
            temp = 0
        sort = temp
        return sort

    def sort_to_flow(self, sort):
        """
        Class restoration computation
        eg: The class is 48, converted to (48+1)*50=1225, then flow=1225
        """
        if sort not in range(self.out_width * self.out_height):
            return -1
        flow = (sort + 1) * 50
        return flow

    # Update the eigenvalues, get the feasible solution set, exchange the current player (out or in)
    # *** Important functions ***

    # @timefn
    def do_flow(self, features_value, time, flow, Tc=5, is_shown=False):
        """
        Perform a traffic output. sort: refers to the type of flow (0~400)
        """
        # self.current_player = 1
        if self.current_player == 1:
            # Update the feature matrix
            features_value = self.update_feature_value_inflow(features_value, time, flow)
            # print("do_flow_sort: {}".format(features_value))

            # Legal solution sets, feasible solutions for special cases
            # inflow
            if time == len(features_value[:, 0]) - 1:
                pass
            else:
                inflow_next = features_value[time + 1, 0]
                inflow_next_sort = self.flow_to_sort(inflow_next)
                self.availables_inflow = [inflow_next_sort]

            # outflow
            if time == 0:
                pass
            else:
                Z1 = features_value[time - 1, 2]
                self.availables_outflow = self.get_available_consider_Z_limit(features_value, flow, Z1)
            # if is_shown:
            #     print(self.availables_outflow)

            # Current player (outflow or inflow)
            self.current_player = 2
            return features_value

        if self.current_player == 2:
            # Update the feature matrix
            features_value = self.update_feature_value_outflow(features_value, time, flow)
            # Current player (outflow or inflow)
            self.current_player = 1

    # @timefn
    def do_inflow(self, features_value, time, flow, Tc=5, is_shown=False):
        """
        Perform a traffic output. sort: refers to the type of flow (0~400)
        """
        # Update the feature matrix
        features_value = self.update_feature_value_inflow(features_value, time, flow)
        # print("do_flow_sort: {}".format(features_value))

        # Legal solution sets, feasible solutions for special cases
        # inflow
        if time == len(features_value[:, 0]) - 1:
            pass
        else:
            inflow_next = features_value[time + 1, 0]
            inflow_next_sort = self.flow_to_sort(inflow_next)
            self.availables_inflow = [inflow_next_sort]

        # outflow
        if time == 0:
            pass
        else:
            Z1 = features_value[time - 1, 2]
            self.availables_outflow = self.get_available_consider_Z_limit(features_value, flow, Z1)
        # if is_shown:
        #     print(self.availables_outflow)

        # Current player (outflow or inflow)
        self.current_player = 2
        return features_value

    # @timefn
    def do_outflow(self, features_value, time, flow, Tc=5, is_shown=False):
        """
        Perform a traffic output. sort: refers to the type of flow (0~400)
        """
        # Update the feature matrix
        features_value = self.update_feature_value_outflow(features_value, time, flow)
        # Current player (outflow or inflow)
        self.current_player = 1
        return features_value

    # @timefn
    def do_flow_playout(self, features_value, time, flow, Tc=5, is_shown=False):
        """
        Perform a traffic output. sort: refers to the type of flow (0~400)
        """
        # self.current_player = 1
        if self.current_player == 1:
            # Update the feature matrix
            features_value = self.update_feature_value_inflow(features_value, time, flow)
            # print("do_flow_sort: {}".format(features_value))

            # Current player (outflow or inflow)
            # inflow
            if time == len(features_value[:, 0]) - 1:
                pass
            else:
                inflow_next = features_value[time + 1, 0]
                inflow_next_sort = self.flow_to_sort(inflow_next)
                self.availables_inflow = [inflow_next_sort]

            # Current player (outflow or inflow)
            self.current_player = 2
            return features_value

        if self.current_player == 2:
            # Update the feature matrix
            features_value = self.update_feature_value_outflow(features_value, time, flow)
            # Current player (outflow or inflow)
            self.current_player = 1
            return features_value

    # @timefn
    def do_inflow_playout(self, features_value, time, flow, Tc=5, is_shown=False):
        """
        Perform a traffic output. sort: refers to the type of flow (0~400)
        """
        # Update the feature matrix
        features_value = self.update_feature_value_inflow(features_value, time, flow)
        # print("do_flow_sort: {}".format(features_value))

        # Legal solution sets, feasible solutions for special cases
        # inflow
        if time == len(features_value[:, 0]) - 1:
            pass
        else:
            inflow_next = features_value[time + 1, 0]
            inflow_next_sort = self.flow_to_sort(inflow_next)
            self.availables_inflow = [inflow_next_sort]

        # Current player (outflow or inflow)
        self.current_player = 2
        return features_value

    # @timefn
    def do_outflow_playout(self, features_value, time, flow, Tc=5, is_shown=False):
        """
        Perform a traffic output. sort: refers to the type of flow (0~400)
        """
        # Update the feature matrix
        features_value = self.update_feature_value_outflow(features_value, time, flow)
        # Current player (outflow or inflow)
        self.current_player = 1
        return features_value

    def get_current_player(self):
        return self.current_player

    def initialize_feature_value(self, inflow, interval_flow, z_up, z_down, flood_constant_feature, is_inflow_all=True, period=100):
        if is_inflow_all:
            features_value = np.zeros((len(inflow), self.features_num))
            features_value[:, 0] = inflow
        else:
            features_value = np.zeros((period, self.features_num))
            features_value[0, 0] = inflow
        features_value[:, 1] = np.NaN
        features_value[0, 2] = z_up
        features_value[0, 3] = z_down
        features_value[0, 10] = flood_constant_feature[0]
        features_value[0, 11] = flood_constant_feature[1]
        features_value[0, 12] = flood_constant_feature[2]
        features_value[0, 13] = flood_constant_feature[3]
        features_value[0, 14] = flood_constant_feature[4]
        features_value[:, 15] = interval_flow
        return features_value

    def update_feature_value_inflow(self, features_value, time, inflow):
        features_value[time, 0] = inflow

        return features_value

    def update_feature_value_outflow(self, features_value, time, outflow):
        features_value[time, 1] = outflow

        # Water level above and below dam
        Z1, Z2 = self.calculate_Z_up_down(features_value, time, outflow)
        features_value[time, 2] = Z1
        features_value[time, 3] = Z2

        # Rate of flood rise and fall in and out of storage
        inflow_rise, inflow_decline = self.calculate_flow_rise_decline(features_value, time, is_inflow=True)
        features_value[0:features_value.shape[0], 4] = inflow_rise
        features_value[0:features_value.shape[0], 5] = inflow_decline

        outflow_rise, outflow_decline = self.calculate_flow_rise_decline(features_value, time, is_inflow=False)
        features_value[0:features_value.shape[0], 6] = outflow_rise
        features_value[0:features_value.shape[0], 7] = outflow_decline

        # Discharge amplitude
        # inflow_variation = self.calculate_flow_variation(features_value, time, variation=50, is_inflow=True)
        # features_value[time, 8] = inflow_variation

        outflow_variation = self.calculate_flow_variation(features_value, time, variation=50, is_inflow=False)
        features_value[time, 8] = outflow_variation

        # Discharge fluctuation limit
        outflow_trend = self.calculate_outflow_trend(features_value, time, Tc=5)

        features_value[time, 9] = outflow_trend

        return features_value

    # Calculate water level above and below dam
    def calculate_Z_up_down(self, features_value, time, outflow):
        if time == 0:
            Z1 = features_value[0, 2]
            Z2 = features_value[0, 3]
            return Z1, Z2
        else:
            V1 = z_to_v(features_value[time - 1, 2])
            V2 = V1 + (features_value[time, 0] - outflow) * 3600 / 10000
            diff_Z = features_value[0, 2] - features_value[0, 3]
            Z1 = v_to_z(V2)
            Z2 = round(Z1 - diff_Z, 2)
            return Z1, Z2

    # Calculate the flood rise and fall rate in and out of storage
    def calculate_flow_rise_decline(self, features_value, time, is_inflow):
        data_inflow = features_value[:, 0]
        if is_inflow:
            data_flow = features_value[:, 0]
        else:
            data_flow = features_value[:, 1]
        flow_rise = np.zeros(features_value.shape[0])
        flow_decline = np.zeros(features_value.shape[0])
        for i in range(time + 1):
            max_inflow = max(data_inflow)
            max_flow_indices = [(i, num) for i, num in enumerate(data_inflow) if num == max_inflow]
            sorted_max_values = sorted(max_flow_indices, key=lambda x: x[0])
            middle_max_value = sorted_max_values[len(sorted_max_values) // 2]

            # The flood growth rate and flood reduction rate are calculated
            if i < middle_max_value[0]:
                flow_rise[i] = round((middle_max_value[1] - data_flow[i]) / (middle_max_value[0] - i), 2)
                flow_decline[i] = 0
            elif i > middle_max_value[0]:
                flow_rise[i] = 0
                flow_decline[i] = round((middle_max_value[1] - data_flow[i]) / (i - middle_max_value[0]), 2)
            else:
                flow_rise[i] = 0
                flow_decline[i] = 0
        return flow_rise, flow_decline

    # Calculate the flow amplitude
    def calculate_flow_variation(self, features_value, time, variation, is_inflow):
        if is_inflow:
            data_flow = features_value[:, 0]
        else:
            data_flow = features_value[:, 1]
        if time == 0:
            return 0
        else:
            flow_variation = data_flow[time] - data_flow[time - 1]
            if flow_variation <= variation:
                return 1
            else:
                return 0

    # Calculate the discharge fluctuation limit
    def calculate_outflow_trend(self, features_value, time, Tc):
        data_outflow = features_value[:, 1]
        trend_list = []
        if time - Tc < 0:
            return 0
        else:
            for i in range(Tc):
                outflow_variation = data_outflow[time - i] - data_outflow[time - (i + 1)]
                if -5 <= outflow_variation <= 5:
                    return 1
                else:
                    if outflow_variation > 0:
                        element = 1
                    else:
                        element = -1
                    trend_list.append(element)
            if all(x == 1 for x in trend_list) or all(x == -1 for x in trend_list):
                return 1
            else:
                return 0

    # Calculating flood control effect
    def calculate_flood_evaluate(self, features_value, return_index=False):
        outflow = features_value[:, 1]
        interval_flow = features_value[:, 15]
        peak_z_up = max(features_value[:, 2])
        low_z_up = min(features_value[:, 2])
        start_z_up = features_value[0, 2]
        end_z_up = features_value[(len(features_value[:, 0]) - 1), 2]
        # print(end_z_up)
        target_z_up = features_value[0, 13]
        flood_high_z_up = features_value[0, 14]

        # Normalized parameter setting
        max_num = 1
        min_num = 0.001
        max_flood_value = 0.987168118085589
        max_flow = 8979
        max_flow_normal = 8979
        min_flow_normal = 0
        max_volume = 53480
        max_volume_normal = 53480
        min_volume_normal = 0
        max_water_level = 21.89
        max_water_level_normal = 21.89
        min_water_level_normal = 0
        flow_weight = 0.373
        volume_weight = 0.320
        water_level_weight = 0.307

        # Calculation index
        # Peak clipping rate
        # peak_reduce_rate = (peak_inflow - peak_outflow) / peak_inflow
        # peak_reduce_rate_normal = (max_num - min_num) * (peak_reduce_rate - 0) / (7792 - 0) + min_num
        # maximum combined flow of flood control point
        combine_flow = outflow + interval_flow
        peak_combine_flow = max(combine_flow)
        peak_reduce_rate_normal = (max_num - min_num) * ((max_flow - peak_combine_flow) - min_flow_normal) / (max_flow_normal - min_flow_normal) + min_num
        # use flood storage
        use_flood_volume = z_to_v(peak_z_up) - z_to_v(start_z_up)
        use_flood_volume_normal = (max_num - min_num) * ((max_volume - use_flood_volume) - min_volume_normal) / (max_volume_normal - min_volume_normal) + min_num
        # the difference between the end water level and the target water level
        diff_z_up = abs(target_z_up - end_z_up)
        diff_z_up_normal = (max_num - min_num) * ((max_water_level - diff_z_up) - min_water_level_normal) / (max_water_level_normal - min_water_level_normal) + min_num

        # calculating flood control effect
        flood_value = round(
            (flow_weight * peak_reduce_rate_normal + volume_weight * use_flood_volume_normal + water_level_weight * diff_z_up_normal)
            / max_flood_value, 4)

        # Return flood control effect value
        if return_index:
            # return flood_value, peak_reduce_rate_normal, use_flood_volume_normal, diff_z_up_normal
            return flood_value, peak_reduce_rate_normal, use_flood_volume_normal, diff_z_up_normal, \
                peak_combine_flow, (max_flow - peak_combine_flow), use_flood_volume, (max_volume - use_flood_volume), \
                diff_z_up, (max_water_level - diff_z_up)
        else:
            return flood_value

    def current_state_auto(self, features_value):
        """
        Returns the current checkerboard state, the input matrix.
        State matrix shape: 24*60*60.
        """
        data = features_value
        in_flow = data[0:, 0]
        out_flow = data[0:, 1]
        z_up = data[0:, 2]
        z_down = data[0:, 3]
        in_flow_rise = data[0:, 4]
        in_flow_reduce = data[0:, 5]
        out_flow_rise = data[0:, 6]
        out_flow_reduce = data[0:, 7]
        out_flow_extent = data[0:, 8]
        out_flow_limit = data[0:, 9]
        reservoir_feature = data[0, 10:13]

        # Initializes the data matrix
        flood_matrix = np.zeros((self.layer, self.in_width, self.in_height))

        out_flow_num = len([flow for flow in out_flow if not np.isnan(flow)])
        if out_flow_num == len(in_flow):
            pass
        else:
            out_flow_num = out_flow_num + 1

        for j in range(out_flow_num):
            a = data_preprocess1.format_data4_flow(in_flow[j])
            if not j == 0:
                b = data_preprocess1.format_data4_flow(out_flow[j - 1])
                d = data_preprocess1.format_data5(z_up[j - 1])
                e = data_preprocess1.format_data5(z_down[j - 1])
                f = data_preprocess1.format_data4_flow(in_flow_rise[j - 1])
                h = data_preprocess1.format_data4_flow(in_flow_reduce[j - 1])
                o = data_preprocess1.format_data4_flow(out_flow_rise[j - 1])
                p = data_preprocess1.format_data4_flow(out_flow_reduce[j - 1])

            m = int((j - 1) // (flood_matrix.shape[2] / 2))
            n = int((j - 1) % (flood_matrix.shape[2] / 2))
            m_1 = int(j // (flood_matrix.shape[2] / 2))
            n_1 = int(j % (flood_matrix.shape[2] / 2))

            if j == 0:
                count = 0
                count = count + 1

                # fill inflow
                if not np.isnan(a).any():
                    for i in range(len(a)):
                        flood_matrix[i + count, a[i], 0] = 1

            else:
                # fill flood frequency（first layer）
                count = 0
                q = data_preprocess1.format_data7(reservoir_feature[0])
                q = q[3:]
                if not np.isnan(q).any():
                    for i in range(flood_matrix.shape[2] // 10):
                        for k in range(len(q)):
                            flood_matrix[count, i * 10:(i + 1) * 10, q[k] + k * 10] = 1
                count = count + 1

                # fill inflow flood and outflow values（2nd-5th layer）
                if not np.isnan(a).any():
                    for i in range(len(a)):
                        flood_matrix[i + count, a[i] + m_1 * 10, n_1 * 2] = 1
                if not np.isnan(b).any():
                    for i in range(len(b)):
                        flood_matrix[i + count, b[i] + m * 10, n * 2 + 1] = 1
                count = count + 4

                # fill all 1 matrix（第6层）
                flood_matrix[count, :, :] = 1
                count = count + 1

                # fill water levels above and below the dam（7th-11th layer）
                if not np.isnan(d).any():
                    for i in range(len(d)):
                        flood_matrix[i + count, d[i] + m * 10, n * 2] = 1
                if not np.isnan(e).any():
                    for i in range(len(e)):
                        flood_matrix[i + count, e[i] + m * 10, n * 2 + 1] = 1
                count = count + 5

                # fill rates of inflow increase and decrease（12th-15th layer）
                if not np.isnan(f).any():
                    for i in range(len(f)):
                        flood_matrix[i + count, f[i] + m * 10, n * 2] = 1
                if not np.isnan(h).any():
                    for i in range(len(h)):
                        flood_matrix[i + count, h[i] + m * 10, n * 2 + 1] = 1
                count = count + 4

                # fill rates of outflow increase and decrease（16th-19th layer）
                if not np.isnan(o).any():
                    for i in range(len(o)):
                        flood_matrix[i + count, o[i] + m * 10, n * 2] = 1
                if not np.isnan(p).any():
                    for i in range(len(p)):
                        flood_matrix[i + count, p[i] + m * 10, n * 2 + 1] = 1
                count = count + 4

                # fill flood peak time（20th layer）
                if not np.isnan(reservoir_feature[1]):
                    flood_matrix[count, :, :] = reservoir_feature[1]
                count = count + 1

                # fill flood shape（21th layer）
                if not np.isnan(reservoir_feature[2]):
                    flood_matrix[count, :, :] = reservoir_feature[2]
                count = count + 1

                # fill limitation of inflow variation（22th layer）
                if not np.isnan(out_flow_extent[j - 1]):
                    flood_matrix[count, m * 10: (m + 1) * 10, (n * 2):(n + 1) * 2] = out_flow_extent[j - 1]
                count = count + 1

                # fill limitation of outflow variation（23th layer）
                if not np.isnan(out_flow_limit[j - 1]):
                    flood_matrix[count, m * 10: (m + 1) * 10, (n * 2):(n + 1) * 2] = out_flow_limit[j - 1]
                count = count + 1

                # fill all 0 matrix（24th layer）
                flood_matrix[count, :, :] = 0

        return flood_matrix

    def current_state_excel(self, file_path, sheet_name):
        """
        Returns the current checkerboard state, the input matrix.
        State matrix shape: 24*60*60.
        """
        square_state = data_preprocess3.generate_dataset3(file_path, True, sheet_name, return_current_board=True)
        return square_state

    def flood_end_time(self, features_value, time, period=100, is_self=False):
        """
        Check whether the flood control scheduling process is complete
        """
        if is_self:
            if time == period - 1:
                return True
            else:
                return False
        else:
            # Total length of time
            T_period = len(features_value[:, 0]) - 1
            # time starts with 0
            if time == T_period:
                return True
            else:
                return False

    def flood_end_count(self, features_value, count, period=100, is_self=False):
        """
        Check whether the flood control scheduling process is complete
        """
        if is_self:
            if count == period * 2:
                return True
            else:
                return False
        else:
            # Total length of count
            T_count = len(features_value[:, 0]) * 2
            # count starts with 0
            if count == T_count:
                return True
            else:
                return False
