import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import share_function

evan, edmund, yumi,\
    friend_1, friend_2, friend_3, friend_4, friend_5, friend_6,\
    friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13\
    = share_function.load_vector_data()

hand_sequence = [(0, 1), (1, 2), (2, 3), (3, 4),
                 (0, 5), (5, 6), (6, 7), (7, 8),
                 (0, 9), (9, 10), (10, 11), (11, 12),
                 (0, 13), (13, 14), (14, 15), (15, 16),
                 (0, 17), (17, 18), (18, 19), (19, 20)]  # 20個向量

pose_sequence = [(0, 12), (0, 11),
                 (12, 14), (14, 16),
                 (11, 13), (13, 15), ]  # 7個向量->6個向量

point_number = len(hand_sequence*2) + len(pose_sequence)


def split_target_evanVersion(new_data_df):
    new_data = new_data_df.to_numpy()
    y = new_data[:, 0]
    x = new_data[:, 1:]
    # y = data[:, 0]
    # x = data[:, 1:]
    # y[y == "salty"] = -1
    # y[y == "snack"] = 1
    return x, y.astype(int)


def split_target(df):
    data = df.to_numpy()
    new_data = np.array(list())
    row_length = 0

    for row in data:
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
    new_data = new_data.reshape(data.shape[0], row_length)
    y = new_data[:, 0]
    x = new_data[:, 1:]
    # y = data[:, 0]
    # x = data[:, 1:]
    # y[y == "salty"] = -1
    # y[y == "snack"] = 1
    return x, y.astype(int)


train_vectors = pd.concat([evan, edmund, friend_1, friend_2, friend_3, friend_4, friend_5,
                          friend_6, friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13])
test_vectors = yumi

evan, edmund, yumi,\
    friend_1, friend_2, friend_3, friend_4, friend_5, friend_6,\
    friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13\
    = share_function.load_point_data()

train_points = pd.concat([evan, edmund, friend_1, friend_2, friend_3, friend_4,
                          friend_5, friend_6, friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13])

train_points = share_function.label_to_float(train_points)

test_points = yumi
test_points = share_function.label_to_float(test_points)


del evan, edmund, yumi, friend_1, friend_2, friend_3, friend_4, friend_5, friend_6,\
    friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13

#! <do shuffle> -> train
train_points, train_vectors = share_function.two_stream_shuffle(
    points=train_points, vectors=train_vectors)

# share_function.two_stream_shuffle(points=train_points, vectors=train_vectors)
# train = train.sample(frac=1).reset_index(drop=True)
# test = test.sample(frac=1).reset_index(drop=True)


x_train_points, y_train_points = \
    split_target_evanVersion(
        train_points)  # origin: x_train, y_train = split_target(train)

x_train_vectors, y_train_vectors = \
    split_target_evanVersion(
        train_vectors)  # origin: x_train, y_train = split_target(train)

x_test_points, y_test_points = \
    split_target_evanVersion(
        test_points)  # origin: x_train, y_train = split_target(train)

x_test_vectors, y_test_vectors = \
    split_target_evanVersion(
        test_vectors)  # origin: x_train, y_train = split_target(train)

# .
# print(x_train.shape)
# x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
# print(x_train.shape)
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
x_train_points = np.asarray(x_train_points).astype(np.float32)
y_train_points = np.asarray(y_train_points).astype(np.float32)

x_train_vectors = np.asarray(x_train_vectors).astype(np.float32)
y_train_vectors = np.asarray(y_train_vectors).astype(np.float32)

x_test_points = np.asarray(x_test_points).astype(np.float32)
y_test_points = np.asarray(y_test_points).astype(np.float32)

x_test_vectors = np.asarray(x_test_vectors).astype(np.float32)
y_test_vectors = np.asarray(y_test_vectors).astype(np.float32)

# x_test = np.asarray(x_test).astype(np.float32)
# y_test = np.asarray(y_test).astype(np.float32)

num_classes = len(np.unique(y_train_vectors))

x_train_points = x_train_points.flatten().reshape(
    x_train_points.shape[0], x_train_points.shape[1]//130, 130)

x_test_points = x_test_points.flatten().reshape(
    x_test_points.shape[0], x_test_points.shape[1]//130, 130)

x_train_vectors = x_train_vectors.flatten().reshape(
    x_train_vectors.shape[0], (x_train_vectors.shape[1]//(point_number*2)), (point_number*2))

x_test_vectors = x_test_vectors.flatten().reshape(
    x_test_vectors.shape[0], (x_test_vectors.shape[1]//(point_number*2)), (point_number*2))

# y_train[y_train == -1] = 0
# y_test[y_test == -1] = 0


model_name = "Lstm"

model = keras.models.load_model(f"{model_name}_best_model.h5")
# confusion matrix
predict_ans = np.argmax(model.predict(
    [x_test_points, x_test_vectors]), axis=-1)  # *  argmax 找最大值的index
cm = tf.math.confusion_matrix(
    y_test_vectors, predict_ans).numpy().astype(np.float32)
print(cm)
print(cm.shape[0])
print(cm.shape[1])

for i in range(cm.shape[0]):
    total_num = 0.0
    for j in range(cm.shape[1]):
        total_num += cm[i][j]
    for j in range(cm.shape[1]):
        cm[i][j] = float(cm[i][j]) / float(total_num)
print(type(cm[0][0]))
df_cm = pd.DataFrame(cm, index=['Salty', 'Snack', 'Bubble Tea',
                                'Dumpling', 'Spicy', 'Sour', 'Sweet', 'Yummy'],
                     columns=['Salty', 'Snack', 'Bubble Tea',
                              'Dumpling', 'Spicy', 'Sour', 'Sweet', 'Yummy'])
fig = plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.show()
# fig.savefig(f'{model_name}_confusion_matrix.png')
