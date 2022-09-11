
import pandas as pd


def shuffle(df):
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    return df_shuffled


def two_stream_shuffle(points, vectors):
    last_col = points.shape[-1]
    temp = pd.concat([points, vectors], axis=1)
    temp = shuffle(df=temp)
    temp1 = temp.iloc[:, :last_col]
    temp2 = temp.iloc[:, last_col:]
    return temp1, temp2


def load_vector_data():
    evan = pd.read_csv("split_data_for_each_one/evan.csv", header=None)
    edmund = pd.read_csv("split_data_for_each_one/edmund.csv", header=None)
    yumi = pd.read_csv("split_data_for_each_one/yumi.csv", header=None)
    # sign_language_df = pd.read_csv(
    #     "Summary_stuff_zero_11st.csv", header=None)
    # print(sign_language_df)

    # sign_language_df = sign_language_df[:][:-2]
    # print(sign_language_df)
    friend_1 = pd.read_csv(
        "split_data_for_each_one/friend_1.csv", header=None)
    friend_2 = pd.read_csv(
        "split_data_for_each_one/friend_2.csv", header=None)
    friend_3 = pd.read_csv(
        "split_data_for_each_one/friend_3.csv", header=None)
    friend_4 = pd.read_csv(
        "split_data_for_each_one/friend_4.csv", header=None)
    friend_5 = pd.read_csv(
        "split_data_for_each_one/friend_5.csv", header=None)
    friend_6 = pd.read_csv(
        "split_data_for_each_one/friend_6.csv", header=None)
    friend_7 = pd.read_csv(
        "split_data_for_each_one/friend_7.csv", header=None)
    friend_8 = pd.read_csv(
        "split_data_for_each_one/friend_8.csv", header=None)
    friend_9 = pd.read_csv(
        "split_data_for_each_one/friend_9.csv", header=None)
    friend_10 = pd.read_csv(
        "split_data_for_each_one/friend_10.csv", header=None)
    friend_11 = pd.read_csv(
        "split_data_for_each_one/friend_11.csv", header=None)
    friend_12 = pd.read_csv(
        "split_data_for_each_one/friend_12.csv", header=None)
    friend_13 = pd.read_csv(
        "split_data_for_each_one/friend_13.csv", header=None)
    return evan, edmund, yumi, friend_1, friend_2, friend_3, friend_4, friend_5, friend_6, friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13


def load_point_data():
    evan = pd.read_csv("nosplit_data_for_each_one/evan.csv", header=None)
    edmund = pd.read_csv("nosplit_data_for_each_one/edmund.csv", header=None)
    yumi = pd.read_csv("nosplit_data_for_each_one/yumi.csv", header=None)
    # sign_language_df = pd.read_csv(
    #     "Summary_stuff_zero_11st.csv", header=None)
    # print(sign_language_df)

    # sign_language_df = sign_language_df[:][:-2]
    # print(sign_language_df)
    friend_1 = pd.read_csv(
        "nosplit_data_for_each_one/friend_1.csv", header=None)
    friend_2 = pd.read_csv(
        "nosplit_data_for_each_one/friend_2.csv", header=None)
    friend_3 = pd.read_csv(
        "nosplit_data_for_each_one/friend_3.csv", header=None)
    friend_4 = pd.read_csv(
        "nosplit_data_for_each_one/friend_4.csv", header=None)
    friend_5 = pd.read_csv(
        "nosplit_data_for_each_one/friend_5.csv", header=None)
    friend_6 = pd.read_csv(
        "nosplit_data_for_each_one/friend_6.csv", header=None)
    friend_7 = pd.read_csv(
        "nosplit_data_for_each_one/friend_7.csv", header=None)
    friend_8 = pd.read_csv(
        "nosplit_data_for_each_one/friend_8.csv", header=None)
    friend_9 = pd.read_csv(
        "nosplit_data_for_each_one/friend_9.csv", header=None)
    friend_10 = pd.read_csv(
        "nosplit_data_for_each_one/friend_10.csv", header=None)
    friend_11 = pd.read_csv(
        "nosplit_data_for_each_one/friend_11.csv", header=None)
    friend_12 = pd.read_csv(
        "nosplit_data_for_each_one/friend_12.csv", header=None)
    friend_13 = pd.read_csv(
        "nosplit_data_for_each_one/friend_13.csv", header=None)
    return evan, edmund, yumi, friend_1, friend_2, friend_3, friend_4, friend_5, friend_6, friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13


def label_to_float(df):
    for i in range(df.shape[0]):
        y = df.iloc[i, 0]
        if y == "salty":
            df.iloc[i, 0] = 0.0
        elif y == "snack":
            df.iloc[i, 0] = 1.0
        elif y == "bubbletea":
            df.iloc[i, 0] = 2.0
        elif y == "dumpling":
            df.iloc[i, 0] = 3.0
        elif y == "spicy":
            df.iloc[i, 0] = 4.0
        elif y == "sour":
            df.iloc[i, 0] = 5.0
        elif y == "sweet":
            df.iloc[i, 0] = 6.0
        elif y == "yummy":
            df.iloc[i, 0] = 7.0
        else:
            df.iloc[i, 0] = 999

    return df
