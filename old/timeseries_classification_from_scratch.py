# -*- coding: utf-8 -*-
"""Timeseries classification from scratch

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1elwjbG0fvtt4BOHiPyahlmP9mYIdv54A

# Timeseries classification from scratch

**Author:** [hfawaz](https://github.com/hfawaz/)<br>
**Date created:** 2020/07/21<br>
**Last modified:** 2021/07/16<br>
**Description:** Training a timeseries classifier from scratch on the FordA dataset from the UCR/UEA archive.
"""

# from google.colab import drive
# drive.mount('/content/drive')

"""## Introduction

This example shows how to do timeseries classification from scratch, starting from raw
CSV timeseries files on disk. We demonstrate the workflow on the FordA dataset from the
[UCR/UEA archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).

## Setup
"""


# myself

# sign_language_df = pd.read_excel("/content/drive/MyDrive/timeseries/Summary_stuff_zero.xlsx")
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
sign_language_df = pd.read_csv(
    "Summary_stuff_zero_9st.csv", header=None)
# print(sign_language_df)

# sign_language_df = sign_language_df[:][:-2]
print(sign_language_df)
# for i in range(308):
#     if(sign_language_df.iat[i, 24311] != 0.0):
#         print("\033[92m here no zero")
#         print("\033[0m")


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


train, test = train_test_split(sign_language_df, test_size=0.2)

x_train, y_train = split_target(train)
x_test, y_test = split_target(test)

# .
print(x_train.shape)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
num_classes = len(np.unique(y_train))

# y_train[y_train == -1] = 0
# y_test[y_test == -1] = 0

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
x_test = np.asarray(x_train).astype(np.float32)
y_test = np.asarray(y_train).astype(np.float32)

"""## Load the data: the FordA dataset

### Dataset description

The dataset we are using here is called FordA.
The data comes from the UCR archive.
The dataset contains 3601 training instances and another 1320 testing instances.
Each timeseries corresponds to a measurement of engine noise captured by a motor sensor.
For this task, the goal is to automatically detect the presence of a specific issue with
the engine. The problem is a balanced binary classification task. The full description of
this dataset can be found [here](http://www.j-wichard.de/publications/FordPaper.pdf).

### Read the TSV data

We will use the `FordA_TRAIN` file for training and the
`FordA_TEST` file for testing. The simplicity of this dataset
allows us to demonstrate effectively how to use ConvNets for timeseries classification.
In this file, the first column corresponds to the label.
"""

# def readucr(filename):
#     data = np.loadtxt(filename, delimiter="\t")
#     y = data[:, 0]
#     x = data[:, 1:]
#     return x, y.astype(int)


# root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

# x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
# x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

# idx = np.random.permutation(len(x_train))
# x_train = x_train[idx]
# y_train = y_train[idx]

"""## Visualize the data

Here we visualize one timeseries example for each class in the dataset.
"""

# classes = np.unique(np.concatenate((y_train, y_test), axis=0))

# plt.figure()
# for c in classes:
#     c_x_train = x_train[y_train == c]
#     plt.plot(c_x_train[0], label="class " + str(c))
# plt.legend(loc="best")
# plt.show()
# plt.close()

"""## Standardize the data

Our timeseries are already in a single length (500). However, their values are
usually in various ranges. This is not ideal for a neural network;
in general we should seek to make the input values normalized.
For this specific dataset, the data is already z-normalized: each timeseries sample
has a mean equal to zero and a standard deviation equal to one. This type of
normalization is very common for timeseries classification problems, see
[Bagnall et al. (2016)](https://link.springer.com/article/10.1007/s10618-016-0483-9).

Note that the timeseries data used here are univariate, meaning we only have one channel
per timeseries example.
We will therefore transform the timeseries into a multivariate one with one channel
using a simple reshaping via numpy.
This will allow us to construct a model that is easily applicable to multivariate time
series.
"""

# x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

"""Finally, in order to use `sparse_categorical_crossentropy`, we will have to count
the number of classes beforehand.
"""

# num_classes = len(np.unique(y_train))

"""Now we shuffle the training set because we will be using the `validation_split` option
later when training.
"""

# idx = np.random.permutation(len(x_train))
# x_train = x_train[idx]
# y_train = y_train[idx]

"""Standardize the labels to positive integers.
The expected labels will then be 0 and 1.
"""

# y_train[y_train == -1] = 0
# y_test[y_test == -1] = 0

"""## Build a model

We build a Fully Convolutional Neural Network originally proposed in
[this paper](https://arxiv.org/abs/1611.06455).
The implementation is based on the TF 2 version provided
[here](https://github.com/hfawaz/dl-4-tsc/).
The following hyperparameters (kernel_size, filters, the usage of BatchNorm) were found
via random search using [KerasTuner](https://github.com/keras-team/keras-tuner).
"""

is_Reshape = False


def make_model(input_shape):
    # ? 跟大小無關 None 甚麼大小都可以 因為globalAveragepooling的設計
    input_layer = keras.layers.Input(shape=input_shape)
    # ? (None, 1)
    conv1 = keras.layers.Conv1D(
        filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    # ? batchnormalization拿掉
    # ? conv1d換成LSTM
    # ? 換成LSTM return seqence 設成true  用LSTM換掉 但LSTM有些參數要設 若沒有return squence不是ture 假設輸入是10個時刻點，只會輸出最後一個時刻點的輸出，因為你這個filter是64個，也就是這整個LSTM跑完之後等於輸出一個向量他是64個軸(維度) 若return sequence為true的話 也就是把input sequence 翻譯成另一個 output sequence 最後做決策的方法可以仿造目前卷基的寫法
    # conv1 = keras.layers.LSTM(units = 64, return_sequences = True)(input_layer)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(
        filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    # conv2 = keras.layers.LSTM(units = 64, return_sequences = True)(conv1)
    # ? 剩下也都這麼做
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(
        filters=64, kernel_size=3, padding="same")(conv2)
    # ? 最後一個LSTM再吐出最後的結果就好
    conv3 = keras.layers.BatchNormalization()(conv3)
    # conv3 = keras.layers.LSTM(units = 64, return_sequences = False)(conv2)
    conv3 = keras.layers.ReLU()(conv3)

    # ? 500 有沒有必要 # 到第三層 500x64 shape -> 1x64 可以改變輸入大小
    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    # ? globalaveragepooling不用寫啦，最後一個LSTM return sequence為false就好了
    # gap = conv3

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


if is_Reshape == False:
    x_train = x_train.flatten().reshape(
        x_train.shape[0], (x_train.shape[1]//(point_number*2)), (point_number*2))
    x_test = x_test.flatten().reshape(
        x_test.shape[0], (x_test.shape[1]//(point_number*2)), (point_number*2))
    is_Reshape = True
print(x_train.shape)

# 2D
# x_train = x_train.flatten().reshape(x_train.shape[0], (x_train.shape[1]//(point_number*2)), (point_number), 2)
# x_test = x_test.flatten().reshape(x_test.shape[0], (x_test.shape[1]//(point_number*2)), (point_number), 2)


model = make_model(input_shape=x_train.shape[1:])
keras.utils.plot_model(model, show_shapes=True)

# 1. 錄影怎麼錄影
# 2. LSTM 網路做上去， 三種典型的序列資料處理技巧
# 3. 標準化座標系
# 4. Grad-cam 找關鍵影像 (讀 keras)

# print(x_train.shape)

# print(x_train)

# print("-"*100)

# # my_x_train = np.array
# # for i in range(len(x_train)):
# x_train = x_train.flatten().reshape(247, (3350//134), 134)

# print(x_train)
# print(x_train.shape[1:])

model.summary()

"""## Train the model


"""

epochs = 500  # 時期
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "Convolution_best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)
# model.save("Convolution_model.h5")


"""## Evaluate model on test data"""

model = keras.models.load_model("Convolution_best_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

"""## Plot the model's training and validation loss"""

metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()

"""We can see how the training accuracy reaches almost 0.95 after 100 epochs.
However, by observing the validation accuracy we can see how the network still needs
training until it reaches almost 0.97 for both the validation and the training accuracy
after 200 epochs. Beyond the 200th epoch, if we continue on training, the validation
accuracy will start decreasing while the training accuracy will continue on increasing:
the model starts overfitting.
"""


# img_array.shape: (1,299,299,3)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(
            last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # print(last_conv_layer_output) # 19偵(1, 19, 64)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel

    # pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # 這19偵的平均

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    # heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


last_conv_layer_name = model.layers[-3].name
# print(model.layers[-3].name)

img_array = x_test[1][tf.newaxis, ...]

heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, pred_index=0)
print(heatmap.shape)  # 19偵
plt.matshow(heatmap)
plt.show()

heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, pred_index=1)
print(heatmap.shape)  # 19偵
plt.matshow(heatmap)
plt.show()
