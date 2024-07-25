# Read the ZV.txt file and arrange the list in ascending order
def read_data_from_file(filename):
    data1 = []
    data2 = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if line:
                values = line.split(",")
                data1.append(float(values[0]))
                data2.append(float(values[1]))

    # both data1 and data2 are sorted, depending on the value of data1
    data1, data2 = zip(*sorted(zip(data1, data2)))

    return data1, data2


# Two points of the water level storage curve interpolation, using V to find Z, and using Z to find V
def insert2(x0, x, y):
    n = len(x)
    if n <= 0:
        return 0
    elif n == 1:
        return y[0]

    i = 0
    while i < n - 1 and x0 > x[i]:
        i += 1

    x1 = x[i - 1]
    x2 = x[i]
    y1 = y[i - 1]
    y2 = y[i]

    y0 = y1 + (x0 - x1) / (x2 - x1) * (y2 - y1)
    return y0


def v_to_z(V0):
    data_Z1, data_V1 = read_data_from_file('Z_V.txt')
    Z0 = round(insert2(V0, data_V1, data_Z1), 2)
    return Z0


def z_to_v(Z0):
    data_Z1, data_V1 = read_data_from_file('Z_V.txt')
    V0 = round(insert2(Z0, data_Z1, data_V1), 2)
    return V0


# Calculate water level above and below dam
def calculate_Z_up_down(inflow, outflow, Z1, diff_Z):
    V1 = z_to_v(Z1)
    V2 = V1 + (inflow - outflow) * 3600 / 10000
    Z1 = v_to_z(V2)
    Z2 = round(Z1 - diff_Z, 2)
    return Z1, Z2


# Example usage
# file_path = './Z_V.txt'  # Change to your file path
# data_Z, data_V = read_data_from_file(file_path)
# print("data list 1:", data_Z)
# print("data list 2:", data_V)
#
# X0 = 234.6
# interpolated_value1 = insert2(X0, data_Z, data_V)
# print("interpolation result:", round(interpolated_value1, 2))
# interpolated_value2 = z_to_v(234.6)
# print("interpolation result:", round(interpolated_value2, 2))
#
#
# Y0 = 1800
# interpolated_value3 = insert2(Y0, data_V, data_Z)
# print("interpolation result:", round(interpolated_value3, 2))
# interpolated_value4 = v_to_z(1800)
# print("interpolation result:", round(interpolated_value4, 2))
