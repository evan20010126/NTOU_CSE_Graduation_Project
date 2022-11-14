from operator import truediv
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


evan = pd.read_csv("nosplit_data_for_each_one/evan.csv", header=None)
print(evan)


def split_target_evanVersion(new_data_df):
    new_data = new_data_df.to_numpy()
    y = new_data[:, 0]
    x = new_data[:, 1:]
    # y = data[:, 0]
    # x = data[:, 1:]
    # y[y == "salty"] = -1
    # y[y == "snack"] = 1
    return x, y.astype(int)


test = evan
# origin: x_test, y_test = split_target(test)

test = test.sample(frac=1).reset_index(drop=True)
# test = test.sample(frac=1).reset_index(drop=True)
x_test, y_test = split_target_evanVersion(test)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)
x_test = x_test.flatten().reshape(
    x_test.shape[0], (x_test.shape[1]//130), 130)

model = keras.models.load_model(
    r"C:\Users\EdmundROG\Desktop\Backup_NTOU_Graduation_Project\backup_experiment\old\points\0\Transformer_best_model.h5")
# confusion matrix
predict_ans = np.argmax(model.predict(
    x_test), axis=-1)  # *  argmax 找最大值的index
cm = tf.math.confusion_matrix(
    y_test, predict_ans).numpy().astype(np.float32)
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

Average = cm

df_cm = pd.DataFrame(cm, index=['Salty', 'Snack', 'Bubble Tea',
                                'Dumpling', 'Spicy', 'Sour', 'Sweet', 'Yummy'],
                     columns=['Salty', 'Snack', 'Bubble Tea',
                              'Dumpling', 'Spicy', 'Sour', 'Sweet', 'Yummy'])
fig = plt.figure(figsize=(10, 7))

print(f"df: {df_cm}")
sn.heatmap(df_cm, annot=True, fmt='.3f')
plt.show()
# fig.savefig(f'{model_name}_confusion_matrix.png')
# fig.savefig(
#     f'C:/Users/User/Desktop/evan_16person_leaveOneOut/{model_name}_confusion_matrix_leave_{leave_person_name}.png')
Average = Average / 16
print("*"*100)
print(Average)
print("*"*100)
