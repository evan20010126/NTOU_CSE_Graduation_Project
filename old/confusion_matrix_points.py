import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
sign_language_df = pd.read_csv(
    "Summary_stuff_zero_8st.csv", header=None)
print(sign_language_df)

hand_sequence = [(0, 1), (1, 2), (2, 3), (3, 4),
                 (0, 5), (5, 6), (6, 7), (7, 8),
                 (0, 9), (9, 10), (10, 11), (11, 12),
                 (0, 13), (13, 14), (14, 15), (15, 16),
                 (0, 17), (17, 18), (18, 19), (19, 20)]  # 20個向量

pose_sequence = [(0, 12), (0, 11),
                 (12, 14), (14, 16),
                 (11, 13), (13, 15), ]  # 7個向量->6個向量

point_number = len(hand_sequence*2) + len(pose_sequence)


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

        # change to vector
        # vector = vector.reshape((vector.shape[0])//130, 65, 2)  # (幾偵, 點, xy)

        # for img in vector:  # 迭代每一偵
        #     pose_points = img[0:23]
        #     left_hand_points = img[23:23+21]
        #     right_hand_points = img[23+21:23+21+21]

        #     for p1, p2 in pose_sequence:
        #         temp_row = np.append(
        #             temp_row, pose_points[p2] - pose_points[p1])

        #     for p1, p2 in hand_sequence:
        #         temp_row = np.append(
        #             temp_row, left_hand_points[p2] - left_hand_points[p1])

        #     for p1, p2 in hand_sequence:
        #         temp_row = np.append(
        #             temp_row, right_hand_points[p2] - right_hand_points[p1])
        # print(temp_row.shape) # 1787
        # changing finish------
        temp_row = np.append(temp_row, vector)  # new
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


train, test = train_test_split(sign_language_df, test_size=0.2)
x_train, y_train = split_target(train)
x_test, y_test = split_target(test)


x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
x_test = np.asarray(x_train).astype(np.float32)
y_test = np.asarray(y_train).astype(np.float32)


# x_train = x_train.flatten().reshape(
#     x_train.shape[0], (x_train.shape[1]//(point_number*2)), point_number*2)
# x_test = x_test.flatten().reshape(
#     x_test.shape[0], (x_test.shape[1]//(point_number*2)), point_number*2)

x_train = x_train.flatten().reshape(
    x_train.shape[0], (x_train.shape[1]//130), 130)
x_test = x_test.flatten().reshape(
    x_test.shape[0], (x_test.shape[1]//130), 130)
model = keras.models.load_model("Transformer_points_best_model.h5")

# confusion matrix
predict_ans = np.argmax(model.predict(x_test), axis=-1)  # *  argmax 找最大值的index
cm = tf.math.confusion_matrix(y_test, predict_ans).numpy().astype(np.float32)
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
fig.savefig('Transformer_points_confusion_matrix.png')
