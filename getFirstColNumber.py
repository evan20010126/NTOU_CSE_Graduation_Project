import pandas as pd


largest_column_count = 0
csv_file = 'Summary_stuff_zero_5st.csv'  # 最新的訓練資料檔
with open(csv_file, 'r') as temp_f:
    lines = temp_f.readlines()
    for l in lines:
        column_count = len(l.split(','))
        if largest_column_count < column_count:
            largest_column_count = column_count
            break
temp_f.close()

print(largest_column_count)
