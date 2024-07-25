# Read the feature.txt file
def read_features_value(filename):
    data1 = []
    data2 = []
    data5 = []

    with open(filename, 'r') as file:
        temp = 0
        for line in file.readlines():
            line = line.strip()
            if line:
                values = line.split(",")
                data1.append(int(values[0]))
                data2.append(int(values[1]))
                if temp == 0:
                    data3 = values[2]
                    data4 = values[3]
                    data5.append(float(values[4]))
                    data5.append(int(values[5]))
                    data5.append(int(values[6]))
                    data5.append(float(values[7]))
                    data5.append(float(values[8]))
                temp += 1

    return data1, data2, data3, data4, data5


# Example usage
# file_path = 'features_value_test.txt'  # Change to your file path
# data_1, data_2, data_3, data_4, data_5 = read_features_value(file_path)
# print("data list 1:", data_1)
# print("data list 2:", data_2)
# print("data list 3:", data_3)
# print("data list 4:", data_4)
# print("data list 5:", data_5)

