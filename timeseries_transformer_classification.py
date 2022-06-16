# -*- coding: utf-8 -*-
# """timeseries_transformer_classification.ipynb

# Automatically generated by Colaboratory.

# Original file is located at
#     https://colab.research.google.com/drive/1ZmfgGaY9XVV2QgS91yWnsPQdHHN6tqz_

# # Timeseries classification with a Transformer model

# **Author:** [Theodoros Ntakouris](https://github.com/ntakouris)<br>
# **Date created:** 2021/06/25<br>
# **Last modified:** 2021/08/05<br>
# **Description:** This notebook demonstrates how to do timeseries classification using a Transformer model.

# ## Introduction

# This is the Transformer architecture from
# [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
# applied to timeseries instead of natural language.

# This example requires TensorFlow 2.4 or higher.

# ## Load the dataset

# We are going to use the same dataset and preprocessing as the
# [TimeSeries Classification from Scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch)
# example.
# """

# from google.colab import drive
# drive.mount('/content/drive')

# import numpy as np
# import pandas as pd

# def readucr(filename):
#     data = np.loadtxt(filename, delimiter="\t")
#     y = data[:, 0]
#     x = data[:, 1:]
#     return x, y.astype(int)

# root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

# x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
# x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

# x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# n_classes = len(np.unique(y_train))

# idx = np.random.permutation(len(x_train))
# x_train = x_train[idx]
# y_train = y_train[idx]

# y_train[y_train == -1] = 0
# y_test[y_test == -1] = 0

from sklearn.model_selection import train_test_split
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import pandas as pd

sign_language_df = pd.read_csv(
    "Summary_stuff_zero_5st.csv")
print(sign_language_df)

# from numpy import genfromtxt

# data = genfromtxt('Summary_stuff_zero_5st.csv', delimiter=',')

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
            temp_row = np.append(temp_row, [-1.0, ])
        elif row[0] == "snack":
            temp_row = np.append(temp_row, [1.0, ])

        # pose: 23個點 left/right:各21個點 23+21*2=65
        vector = row[1:]
        vector = vector.reshape((vector.shape[0])//130, 65, 2)  # (幾偵, 點, xy)
        for img in vector:  # 迭代每一偵
            pose_points = img[0:23]
            left_hand_points = img[23:23+21]
            right_hand_points = img[23+21:23+21+21]

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


# def split_target(data_df):
#   data_np = data_df.to_numpy()
#   y = data_np[:, 0]
#   x = data_np[:, 1:]
#   x = transfer_to_vector(x)
#   y[y == 'salty'] = 0
#   y[y == 'snack'] = 1
#   return x, y

# 為了解決ResourceExhaustedError的暫時處理方法:
# def reduce_data(data_df):
#   new_df = data_df.iloc[:,:501]
#   return new_df
#re_sign_language_df = reduce_data(sign_language_df)
train, test = train_test_split(sign_language_df, test_size=0.2)
x_train, y_train = split_target(train)
x_test, y_test = split_target(test)

# print("len(x_train): ",len(x_train))
# print("x_train.shape[1]: ",x_train.shape[1])

x_train = x_train.flatten().reshape(
    x_train.shape[0], (x_train.shape[1]//(point_number*2)), point_number*2)
x_test = x_test.flatten().reshape(
    x_test.shape[0], (x_test.shape[1]//(point_number*2)), point_number*2)


print("x_train[0]:\n", x_train[0])
print("y_train[0]:\n", y_train[0])

# x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

n_classes = len(np.unique(y_train))

idx = np.random.permutation(len(x_train))
idx_y = np.random.permutation(len(y_train))
x_train = x_train[idx]
y_train = y_train[idx_y]

# Fail to convert a NumPy array to Tensor:
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
x_test = np.asarray(x_train).astype(np.float32)
y_test = np.asarray(y_train).astype(np.float32)

"""## Build the model

Our model processes a tensor of shape `(batch size, sequence length, features)`,
where `sequence length` is the number of time steps and `features` is each input
timeseries.

You can replace your classification RNN layers with this one: the
inputs are fully compatible!
"""


"""We include residual connections, layer normalization, and dropout.
The resulting layer can be stacked multiple times.

The projection layers are implemented through `keras.layers.Conv1D`.
"""


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    # x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    # x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


"""The main part of our model is now complete. We can stack multiple of those
`transformer_encoder` blocks and we can also proceed to add the final
Multi-Layer Perceptron classification head. Apart from a stack of `Dense`
layers, we need to reduce the output tensor of the `TransformerEncoder` part of
our model down to a vector of features for each data point in the current
batch. A common way to achieve this is to use a pooling layer. For
this example, a `GlobalAveragePooling1D` layer is sufficient.
"""


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


"""## Train and evaluate"""

input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=64,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.2,
    dropout=0.25,
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True)]
# callbacks = [
#     keras.callbacks.ModelCheckpoint(
#         "best_model.h5", save_best_only=True, monitor="sparse_categorical_accuracy"
#     ),
#     keras.callbacks.ReduceLROnPlateau(
#         monitor="sparse_categorical_accuracy", factor=0.5, patience=20, min_lr=0.0001
#     ),
#     keras.callbacks.EarlyStopping(monitor="sparse_categorical_accuracy", patience=50, verbose=1),
# ]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,  # 64
    callbacks=callbacks,
)

model.save('my_model_test.h5')

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print("Test Accuracy:", test_acc)
print("Test loss:", test_loss)

"""## Conclusions

In about 110-120 epochs (25s each on Colab), the model reaches a training
accuracy of ~0.95, validation accuracy of ~84 and a testing
accuracy of ~85, without hyperparameter tuning. And that is for a model
with less than 100k parameters. Of course, parameter count and accuracy could be
improved by a hyperparameter search and a more sophisticated learning rate
schedule, or a different optimizer.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/timeseries_transformer_classification) and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/timeseries_transformer_classification).
"""


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
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # grads = grads[tf.newaxis,...]
    # grads = grads[tf.newaxis,...]
    # print(grads)
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    # pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation

    last_conv_layer_output = last_conv_layer_output[0]
    # last_conv_layer_output = last_conv_layer_output
    # print(last_conv_layer_output)
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    # heatmap = tf.squeeze(heatmap)
    # print(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


print(model.layers[-5].name)
last_conv_layer_name = model.layers[-5].name

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
