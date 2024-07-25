import pandas as pd
import numpy as np
import torch

import time
from functools import wraps


def timefn(fn):
    """
    Modifiers to calculate performance
    """

    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1: .5f} s")
        return result

    return measure_time


# Decimal point for 1, 2, 3
xsw1 = 0
xsw2 = 0
xsw3 = 0


# Process the number into 4 digits
def format_data4_flow(number):
    if np.isnan(number):
        return number
    else:
        qw = int(number // 1000)
        bw = int(number // 100 % 10)
        sw = int(number // 10 % 10)
        gw = int(number % 10)
        return qw, bw, sw, gw


def format_data4_rain(number):
    global xsw1, xsw2
    if np.isnan(number):
        return number
    else:
        qw = int(number // 1000)
        bw = int(number // 100 % 10)
        sw = int(number // 10 % 10)
        gw = int(number % 10)
        xsw1 = int(number * 10 % 10)
        xsw2 = int(number * 100 % 10)
        return sw, gw, xsw1, xsw2


# Process the number into 5 digits
def format_data5(number):
    global xsw1, xsw2
    if np.isnan(number):
        return number
    else:
        qw = int(number // 1000)
        bw = int(number // 100 % 10)
        sw = int(number // 10 % 10)
        gw = int(number % 10)
        xsw1 = int(number * 10 % 10)
        xsw2 = int(number * 100 % 10)
        if number >= 1000:
            if number - qw * 1000 - bw * 100 - sw * 10 - gw == 0:
                return qw, bw, sw, gw, 0
            else:
                return qw, bw, sw, gw, xsw1
        elif number >= 100:
            if number - bw * 100 - sw * 10 - gw == 0:
                return bw, sw, gw, 0, 0
            else:
                return bw, sw, gw, xsw1, xsw2
        elif number >= 10:
            if number - sw * 10 - gw == 0:
                return 0, sw, gw, 0, 0
            else:
                return 0, sw, gw, xsw1, xsw2
        else:
            if number - gw == 0:
                return 0, 0, gw, 0, 0
            else:
                return 0, 0, gw, xsw1, xsw2


# Process the number into 6 digits
def format_data6(number):
    global xsw1, xsw2
    if np.isnan(number):
        return number
    else:
        qw = int(number // 1000)
        bw = int(number // 100 % 10)
        sw = int(number // 10 % 10)
        gw = int(number % 10)
        xsw1 = int(number * 10 % 10)
        xsw2 = int(number * 100 % 10)
        if number >= 1000:
            if number - qw * 1000 - bw * 100 - sw * 10 - gw == 0:
                return qw, bw, sw, gw, 0, 0
            else:
                return qw, bw, sw, gw, xsw1, xsw2
        elif number >= 100:
            if number - bw * 100 - sw * 10 - gw == 0:
                return 0, bw, sw, gw, 0, 0
            else:
                return 0, bw, sw, gw, xsw1, xsw2
        elif number >= 10:
            if number - sw * 10 - gw == 0:
                return 0, 0, sw, gw, 0, 0
            else:
                return 0, 0, sw, gw, xsw1, xsw2
        else:
            if number - gw == 0:
                return 0, 0, 0, gw, 0, 0
            else:
                return 0, 0, 0, gw, xsw1, xsw2


# Process the number into 7 digits
def format_data7(number):
    global xsw1, xsw2, xsw3
    if np.isnan(number):
        return number
    else:
        qw = int(number // 1000)
        bw = int(number // 100 % 10)
        sw = int(number // 10 % 10)
        gw = int(number % 10)
        xsw1 = int(number * 10 % 10)
        xsw2 = int(number * 100 % 10)
        xsw3 = int(number * 1000 % 10)
        if number >= 1000:
            if number - qw * 1000 - bw * 100 - sw * 10 - gw == 0:
                return qw, bw, sw, gw, 0, 0, 0
            else:
                return qw, bw, sw, gw, xsw1, xsw2, xsw3
        elif number >= 100:
            if number - bw * 100 - sw * 10 - gw == 0:
                return 0, bw, sw, gw, 0, 0, 0
            else:
                return 0, bw, sw, gw, xsw1, xsw2, xsw3
        elif number >= 10:
            if number - sw * 10 - gw == 0:
                return 0, 0, sw, gw, 0, 0, 0
            else:
                return 0, 0, sw, gw, xsw1, xsw2, xsw3
        else:
            if number - gw == 0:
                return 0, 0, 0, gw, 0, 0, 0
            else:
                return 0, 0, 0, gw, xsw1, xsw2, xsw3
