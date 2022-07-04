import pandas as pd
import math
from numpy import genfromtxt
import csv

max_column = 24311


def write_csv(file_name, all_data):
    print("Writing csv...")

    # all_data.insert(0, "webcam")
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(all_data)

    print("\Finish writing csv/")


def preprocess():
    new_row = list()
    largest_column_count = 0
    csv_file = 'webcam.csv'
    with open(csv_file, 'r') as temp_f:
        lines = temp_f.readlines()
        for l in lines:
            column_count = len(l.split(','))
            if largest_column_count < column_count:
                largest_column_count = column_count
    temp_f.close()

    # print(largest_column_count)

    frame_cutting = math.ceil(
        ((largest_column_count-1)/130.0) / ((max_column-1)/130.0))
    print(frame_cutting)
    data = genfromtxt('webcam.csv', delimiter=',',)  # numpy讀csv
    print(data.shape)
    new_row.append(data[0])
    count_column = 1
    for i in range(1, largest_column_count, frame_cutting * 130):
        for j in range(130):
            new_row.append(data[i+j])
            count_column += 1
    for i in range(max_column - count_column):
        new_row.append(0)

    write_csv("webcam_stuff_zero.csv", new_row)
    print(len(new_row))
    # stuff 0
    # column_names = [i for i in range(0, largest_column_count)]

    # # print(column_names[-1])

    # data = pd.read_csv(csv_file, delimiter=",",
    #                 header=None, encoding='utf8', names=column_names, engine='python')


    # data = data.fillna(0.0)
    # data.to_csv('Summary_stuff_zero_5st.csv', index=False, header=False)
# preprocess()
