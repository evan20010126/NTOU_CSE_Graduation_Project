import seaborn as sn
import pyscreenshot as ImageGrab
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
abcdefg = True
sys.path.append(".")
# sys.path.append("..")
if abcdefg:
    import old.share_function as share_function


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


model_list = [
    "auto_leave_person/0/Lstm_best_model.h5",
    "auto_leave_person/1/Lstm_best_model.h5",
    "auto_leave_person/2/Lstm_best_model.h5",
    "auto_leave_person/3/Lstm_best_model.h5",
    "auto_leave_person/4/Lstm_best_model.h5",
    "auto_leave_person/5/Lstm_best_model.h5",
    "auto_leave_person/6/Lstm_best_model.h5",
    "auto_leave_person/7/Lstm_best_model.h5",
    "auto_leave_person/8/Lstm_best_model.h5",
    "auto_leave_person/9/Lstm_best_model.h5",
    "auto_leave_person/10/Lstm_best_model.h5",
    "auto_leave_person/11/Lstm_best_model.h5",
    "auto_leave_person/12/Lstm_best_model.h5",
    "auto_leave_person/13/Lstm_best_model.h5",
    "auto_leave_person/14/Lstm_best_model.h5",
    "auto_leave_person/15/Lstm_best_model.h5",
]

# -- point --
evan, edmund, yumi,\
    friend_1, friend_2, friend_3, friend_4, friend_5, friend_6,\
    friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13\
    = share_function.load_point_data()

all_person_pd_points = [evan, yumi, edmund, friend_1, friend_2, friend_3, friend_4, friend_5,
                        friend_6, friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13]

del evan, edmund, yumi, friend_1, friend_2, friend_3, friend_4, friend_5, friend_6,\
    friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13


def split_target_evanVersion(new_data_df):
    new_data = new_data_df.to_numpy()
    y = new_data[:, 0]
    x = new_data[:, 1:]
    # y = data[:, 0]
    # x = data[:, 1:]
    # y[y == "salty"] = -1
    # y[y == "snack"] = 1
    return x, y.astype(int)


test_points = -999

###
Average = None
avg_first = True
###

for leave_idx in range(16):
    model = model_list[leave_idx]
    test_points = all_person_pd_points[leave_idx]
    test_points = share_function.label_to_float(test_points)

    x_test_points, y_test_points = split_target_evanVersion(test_points)

    x_test_points = np.asarray(x_test_points).astype(np.float32)
    y_test_points = np.asarray(y_test_points).astype(np.float32)

    x_test_points = x_test_points.flatten().reshape(
        x_test_points.shape[0], x_test_points.shape[1]//130, 130)

    model = keras.models.load_model(model)

    # confusion matrix
    predict_ans = np.argmax(model.predict(
        x_test_points), axis=-1)  # *  argmax 找最大值的index
    cm = tf.math.confusion_matrix(
        y_test_points, predict_ans).numpy().astype(np.float32)
    # print(cm)
    # print(cm.shape[0])
    # print(cm.shape[1])

    for i in range(cm.shape[0]):
        total_num = 0.0
        for j in range(cm.shape[1]):
            total_num += cm[i][j]
        for j in range(cm.shape[1]):
            cm[i][j] = float(cm[i][j]) / float(total_num)
    print(type(cm[0][0]))
    print(cm)
    if avg_first:
        Average = cm
        avg_first = False
    else:
        Average = Average + cm

# del all_person_pd_vector
del all_person_pd_points


Average = Average / 16
print("*"*100)
print(Average)
print("*"*100)

df_avg = pd.DataFrame(Average, index=['Salty', 'Snack', 'Bubble Tea',
                                      'Dumpling', 'Spicy', 'Sour', 'Sweet', 'Yummy'],
                      columns=['Salty', 'Snack', 'Bubble Tea',
                               'Dumpling', 'Spicy', 'Sour', 'Sweet', 'Yummy'])
fig = plt.figure(figsize=(10, 7))

# print(f"df: {df_avg}")
sn.heatmap(df_avg, annot=True, fmt='.3f')
plt.show()
fig.savefig("auto_leave_person/avg_confusion_matrix.png")
