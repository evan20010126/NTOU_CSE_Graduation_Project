import pandas as pd
import sys
import os
START = True
if START:
    p = os.getcwd()
    sys.path.append(p)
    import preprocess_userCSV


# chunk_list = list()
# data_chunks = pd.read_csv(
#     f'Summary_{summary_file_version}st.csv', chunksize=10)
# for data_chunk in data_chunks:
#     data_chunk = data_chunk.fillna(0.0)
#     chunk_list.append(data_chunk)

# print(chunk_list)


# 00957202
# df = pd.read_csv('Summary_5st.csv', delimiter=",",
#                  header=None, encoding='utf8', engine='python')

# df = df.fillna(0.0)

# print(df)

# df.to_csv(
#     'Summary_stuff_zero_5st.csv', index=False, header=False)

# 00853029

summary_file_version = "9_1"  # !改這裡就好
largest_column_count = 0  # !還有這裡
csv_file = f'output.csv'
row_num = 0
with open(csv_file, 'r') as temp_f:
    lines = temp_f.readlines()
    row_num = len(lines)
    # for l in lines:
    #     column_count = len(l.split(','))
    #     if largest_column_count < column_count:
    #         largest_column_count = column_count
temp_f.close()

# print(largest_column_count)

# column_names = [i for i in range(0, largest_column_count)]

# # print(column_names[-1])

# data = pd.read_csv(csv_file, delimiter=",",
#                    header=None, encoding='utf8', names=column_names, engine='python')

# data = data.fillna(0.0)
for i in range(row_num):
    preprocess_userCSV.preprocess(max_column=largest_column_count, src_csv_file=csv_file,
                                  dest_csv_file=f'Summary_stuff_zero_{summary_file_version}st.csv', process_row=i, mode='a')

# data.to_csv(
#     f'Summary_stuff_zero_{summary_file_version}st.csv', index=False, header=False)
