import pandas as pd

excel_files = [r'C:\Users\Longwen-Liu\Desktop\FloodGo-Version1.0-main\data\Test flood data.xlsx']  # Change to your file path
output_file = r'C:\APP\Python\FloodGo-V1.0\flood_MCTS\flood_data\features_value_0.2%.txt'  # Change to your file path

output_data = []

for file in excel_files:
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, "20100618(0.8)")  # Change to specified sheet of Excel
    column1_data = df.iloc[1:, 0].tolist()
    column2_data = df.iloc[1:, 26].tolist()

    data1 = df.iloc[0, 0]
    data2 = df.iloc[0, 26]
    data3 = df.iloc[0, 3]
    data4 = df.iloc[0, 4]
    data5 = df.iloc[0, 14]
    data6 = int(df.iloc[18, 14])
    data7 = int(df.iloc[19, 14])
    data8 = df.iloc[10, 16]
    data9 = df.iloc[11, 16]

    max_len = max(len(column1_data), len(column2_data))
    column1_data = column1_data[:max_len]
    column2_data = column2_data[:max_len]

    for i in range(max_len):
        output_data.append((column1_data[i], column2_data[i]))

output_data.append((data1, data2, data3, data4, data5, data6, data7, data8, data9))

with open(output_file, 'w') as f:
    f.write(','.join(map(str, output_data[-1])) + '\n')
    for item in output_data[:-1]:
        f.write(','.join(map(str, item)) + '\n')
