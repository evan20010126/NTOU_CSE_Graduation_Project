import pandas as pd
import math
from numpy import genfromtxt
import csv
import numpy as np


def write_csv(file_name, all_data, mode='w'):
    print("Writing csv...")

    # all_data.insert(0, "webcam")
    with open(file_name, mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerow(all_data)

    print("\Finish writing csv/")


def preprocess(max_column=0, src_csv_file='webcam.csv', dest_csv_file="webcam_stuff_zero.csv", process_row=0, mode='w'):
    if (max_column == 0):
        print("There is no column in your video.")
    else:
        new_row = list()
        largest_column_count = 0
        with open(src_csv_file, 'r') as temp_f:
            lines = temp_f.readlines()
            for l in lines:
                column_count = len(l.split(','))
                if largest_column_count < column_count:
                    largest_column_count = column_count
        temp_f.close()

        # print(largest_column_count)

        frame_cutting = math.ceil(
            ((largest_column_count-1)/130.0) / ((max_column-1)/130.0))
        print(f"frame_compact(frame_cutting):{frame_cutting}")

        # data = genfromtxt(src_csv_file, delimiter=',',)  # numpy讀csv

        with open(src_csv_file, newline='') as f:
            lines = f.readlines()
            data = lines[process_row].split(',')
            sign_label = data[0]
            data = np.array(data[1:]).astype(np.float32)
            data = np.insert(data, 0, -123456.0)

        new_row.append(sign_label)  # append label
        count_column = 1
        # cutting frame
        for i in range(1, len(data), frame_cutting * 130):
            for j in range(130):
                new_row.append(data[i+j])
                count_column += 1
        # for i in range(max_column - count_column):
            # new_row.append(0)

        stuffed_list = np.array(list())
        stuffed_list = np.append(stuffed_list, [new_row[0]])
        new_row = np.array(new_row[1:])
        new_row = new_row.reshape(-1, 130)
        step = count_column/max_column
        # for(int i = 0;i<=max_column;i++)
        i = -step

        # print(f'max_column : {max_column}')

        while(0.001 <= (max_column-1)/130 - i):

            if (len(stuffed_list)-1)/130 == (max_column-1)/130:
                break
            # print(f'counter = {counter}')
            if math.floor(i) != math.floor(i+step):
                carry = True
            else:
                carry = False
            print(f'max_column-1)/130 : {(max_column-1)/130}')
            i = i+step
            print(f'i = {i}')
            if carry:
                stuffed_list = np.append(stuffed_list, new_row[math.floor(i)])
            else:
                stuffed_list = np.append(stuffed_list, new_row[math.floor(i)])

        stuffed_list = stuffed_list.flatten()
        write_csv(dest_csv_file, stuffed_list, mode)
        print(len(stuffed_list))
        # stuff 0
        # column_names = [i for i in range(0, largest_column_count)]

        # # print(column_names[-1])

        # data = pd.read_csv(csv_file, delimiter=",",
        #                 header=None, encoding='utf8', names=column_names, engine='python')

        # data = data.fillna(0.0)
        # data.to_csv('Summary_stuff_zero_5st.csv', index=False, header=False)
# preprocess(29251)
