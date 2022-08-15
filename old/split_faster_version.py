from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# myself
hand_sequence = [(0, 1), (1, 2), (2, 3), (3, 4),
                 (0, 5), (5, 6), (6, 7), (7, 8),
                 (0, 9), (9, 10), (10, 11), (11, 12),
                 (0, 13), (13, 14), (14, 15), (15, 16),
                 (0, 17), (17, 18), (18, 19), (19, 20)]  # 20個向量

pose_sequence = [(0, 12), (0, 11),
                 (12, 14), (14, 16),
                 (11, 13), (13, 15), ]  # 7個向量->6個向量

point_number = len(hand_sequence*2) + len(pose_sequence)


def to_vector_evanVersion(df):

    data = df.to_numpy()
    new_data = np.array(list())
    row_length = 0

    Progress_bar_counter_now = 0
    Progress_bar_counter_max = len(data)-1

    for row in data:
        print("\r", end="")
        print(f"row: {Progress_bar_counter_now}/{Progress_bar_counter_max} " +
              int(Progress_bar_counter_now/Progress_bar_counter_max*10)*"▓", end="")
        sys.stdout.flush()
        Progress_bar_counter_now += 1
        temp_row = np.array(list())

        if row[0] == "salty":
            temp_row = np.append(temp_row, [0.0, ])
        elif row[0] == "snack":
            temp_row = np.append(temp_row, [1.0, ])
        elif row[0] == "bubbletea":
            temp_row = np.append(temp_row, [2.0, ])
        elif row[0] == "dumpling":
            temp_row = np.append(temp_row, [3.0, ])
        elif row[0] == "spicy":
            temp_row = np.append(temp_row, [4.0, ])
        elif row[0] == "sour":
            temp_row = np.append(temp_row, [5.0, ])
        elif row[0] == "sweet":
            temp_row = np.append(temp_row, [6.0, ])
        elif row[0] == "yummy":
            temp_row = np.append(temp_row, [7.0, ])

        # pose: 23個點 left/right:各21個點 23+21*2=65
        vector = row[1:]
        vector = vector.reshape((vector.shape[0])//130, 65, 2)  # (幾偵, 點, xy)
        for img in vector:  # 迭代每一偵
            pose_points = img[0:23]
            left_hand_points = img[23:23+21]
            right_hand_points = img[23+21:23+21+21]

            # # 加上不是向量的點
            # temp_row = np.append(temp_row, pose_points[0])
            # temp_row = np.append(temp_row, pose_points[15])
            # temp_row = np.append(temp_row, pose_points[16])
            # #

            for p1, p2 in pose_sequence:
                temp_row = np.append(
                    temp_row, pose_points[p2] - pose_points[p1])

            for p1, p2 in hand_sequence:
                temp_row = np.append(
                    temp_row, left_hand_points[p2] - left_hand_points[p1])

            for p1, p2 in hand_sequence:
                temp_row = np.append(
                    temp_row, right_hand_points[p2] - right_hand_points[p1])
        # print(temp_row.shape) # 1787
        row_length = temp_row.shape[0]
        new_data = np.append(new_data, temp_row)
    print("\n")
    new_data = new_data.reshape(data.shape[0], row_length)
    return new_data


evan = pd.read_csv(
    "Summary_stuff_zero_11_1st.csv", header=None)
edmund = pd.read_csv(
    "Summary_stuff_zero_11_2st.csv", header=None)
yumi = pd.read_csv(
    "Summary_stuff_zero_11_3st.csv", header=None)

friend_1 = pd.read_csv(
    "../edmund_friends/output_media_1_stuff_zero.csv", header=None)
friend_2 = pd.read_csv(
    "../edmund_friends/output_media_2_stuff_zero.csv", header=None)
friend_3 = pd.read_csv(
    "../edmund_friends/output_media_3_stuff_zero.csv", header=None)
friend_4 = pd.read_csv(
    "../edmund_friends/output_media_4_stuff_zero.csv", header=None)
friend_5 = pd.read_csv(
    "../evan_friends/output_big_stuff_zero.csv", header=None)
friend_6 = pd.read_csv(
    "../evan_friends/output_bingbing_stuff_zero.csv", header=None)
friend_7 = pd.read_csv(
    "../evan_friends/output_chen_stuff_zero.csv", header=None)
friend_8 = pd.read_csv(
    "../evan_friends/output_Chiayi_stuff_zero.csv", header=None)
friend_9 = pd.read_csv(
    "../evan_friends/output_pich_stuff_zero.csv", header=None)
friend_10 = pd.read_csv(
    "../yumi_friends/output_Howard_stuff_zero.csv", header=None)
friend_11 = pd.read_csv(
    "../yumi_friends/output_justin_stuff_zero.csv", header=None)
friend_12 = pd.read_csv(
    "../yumi_friends/output_me_stuff_zero.csv", header=None)
friend_13 = pd.read_csv(
    "../yumi_friends/output_other_stuff_zero.csv", header=None)

all_person = \
    [evan, yumi, edmund, friend_1, friend_2, friend_3, friend_4, friend_5,
     friend_6, friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13]

file_name_list = \
    ["evan", "yumi", "edmund", "friend_1", "friend_2", "friend_3", "friend_4", "friend_5",
             "friend_6", "friend_7", "friend_8", "friend_9", "friend_10", "friend_11", "friend_12", "friend_13"]
# split_data_for_each_one
for i in range(len(all_person)):
    print(file_name_list[i])
    new_data = to_vector_evanVersion(all_person[i])
    pd.DataFrame(new_data).to_csv(
        f'split_data_for_each_one\{file_name_list[i]}.csv', index=False, header=False)
