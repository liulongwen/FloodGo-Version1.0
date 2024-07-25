import d2l.torch
import pandas as pd
import numpy as np
from d2l import torch as d2l
import labels_restore as rs


# Decimal point for 1, 2
xsw1 = 0
xsw2 = 0


# Process the number into 4 digits
def format_data4(number):
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
                return qw, bw, sw, gw
            else:
                return False
        elif number >= 100:
            if number - bw * 100 - sw * 10 - gw == 0:
                return 0, bw, sw, gw
            else:
                return False
        elif number >= 10:
            if number - sw * 10 - gw == 0:
                return 0, 0, sw, gw
            else:
                return sw, gw, xsw1, xsw2
        else:
            if number - gw == 0:
                return 0, 0, 0, gw
            else:
                return 0, gw, xsw1, xsw2
