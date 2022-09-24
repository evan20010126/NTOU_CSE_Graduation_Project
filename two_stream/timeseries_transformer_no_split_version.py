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
import sys
import seaborn as sn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import pandas as pd
abcdefg = True
sys.path.append(".")
# sys.path.append("..")
if abcdefg:
    import old.share_function as share_function

evan, edmund, yumi,\
    friend_1, friend_2, friend_3, friend_4, friend_5, friend_6,\
    friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13\
    = share_function.load_vector_data()

train_vectors = pd.concat([evan, yumi, edmund, friend_1, friend_3, friend_4,
                           friend_5, friend_6, friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13])
test_vectors = friend_2

evan, edmund, yumi,\
    friend_1, friend_2, friend_3, friend_4, friend_5, friend_6,\
    friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13\
    = share_function.load_point_data()

train_points = pd.concat([evan, yumi, edmund, friend_1, friend_3, friend_4,
                          friend_5, friend_6, friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13])
train_points = share_function.label_to_float(train_points)
test_points = friend_2
test_points = share_function.label_to_float(test_points)

del evan, edmund, yumi, friend_1, friend_2, friend_3, friend_4, friend_5, friend_6,\
    friend_7, friend_8, friend_9, friend_10, friend_11, friend_12, friend_13

#! <do shuffle> -> train
train_points, train_vectors = share_function.two_stream_shuffle(
    points=train_points, vectors=train_vectors)

# share_function.two_stream_shuffle(points=train_points, vectors=train_vectors)
# train = train.sample(frac=1).reset_index(drop=True)
# test = test.sample(frac=1).reset_index(drop=True)


def split_target_evanVersion(new_data_df):
    new_data = new_data_df.to_numpy()
    y = new_data[:, 0]
    x = new_data[:, 1:]
    # y = data[:, 0]
    # x = data[:, 1:]
    # y[y == "salty"] = -1
    # y[y == "snack"] = 1
    return x, y.astype(int)


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


x_train_points = np.asarray(x_train_points).astype(np.float32)
y_train_points = np.asarray(y_train_points).astype(np.float32)

x_train_vectors = np.asarray(x_train_vectors).astype(np.float32)
y_train_vectors = np.asarray(y_train_vectors).astype(np.float32)

x_test_points = np.asarray(x_test_points).astype(np.float32)
y_test_points = np.asarray(y_test_points).astype(np.float32)

x_test_vectors = np.asarray(x_test_vectors).astype(np.float32)
y_test_vectors = np.asarray(y_test_vectors).astype(np.float32)


x_train_points = x_train_points.flatten().reshape(
    x_train_points.shape[0], x_train_points.shape[1]//130, 130)

x_test_points = x_test_points.flatten().reshape(
    x_test_points.shape[0], x_test_points.shape[1]//130, 130)

x_train_vectors = x_train_vectors.flatten().reshape(
    x_train_vectors.shape[0], (x_train_vectors.shape[1]//(point_number*2)), (point_number*2))

x_test_vectors = x_test_vectors.flatten().reshape(
    x_test_vectors.shape[0], (x_test_vectors.shape[1]//(point_number*2)), (point_number*2))

n_classes = len(np.unique(y_train_vectors))

# idx = np.random.permutation(len(x_train))
# idx_y = np.random.permutation(len(y_train))
# x_train = x_train[idx]
# y_train = y_train[idx_y]

# Fail to convert a NumPy array to Tensor:


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
    # x = keras.layers.BatchNormalization()(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    # x = layers.LayerNormalization(epsilon=1e-6)(res)
    # x = keras.layers.BatchNormalization()(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


"""The main part of our model is now complete. We can stack multiple of those
`transformer_encoder` blocks and we can also proceed to add the final
Multi-Layer Perceptron classification head. Apart from a stack of `Dense`
layers, we need to reduce the output tensor of the `TransformerEncoder` part of
our model down to a vector of features for each data point in the current
batch. A common way to achieve this is to use a pooling layer. For
this example, a `GlobalAveragePooling1D` layer is sufficient.
"""


def share_stream(x_shape, num_transformer_blocks, head_size, num_heads, ff_dim, dropout):
    x = keras.Input(shape=x_shape)
    # conv
    conv1 = keras.layers.Conv1D(
        filters=32, kernel_size=3, padding="same")(x)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv2 = keras.layers.Conv1D(
        filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    xd = layers.Dropout(0.25)(conv2)
    ###########

    for _ in range(num_transformer_blocks):
        xd = transformer_encoder(xd, head_size, num_heads, ff_dim, dropout)

    last = layers.GlobalAveragePooling1D(data_format="channels_first")(
        xd)  # data_format="channels_first"
    shared_layer = keras.models.Model(x, last)
    return shared_layer


def build_model(
    input_shape_vector,
    input_shape_point,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):

    inputs_point = keras.layers.Input(shape=input_shape_point)
    inputs_vector = keras.layers.Input(shape=input_shape_vector)
    # x = inputs

    point_stream = share_stream(
        input_shape_point, num_transformer_blocks, head_size, num_heads, ff_dim, dropout)
    vector_stream = share_stream(
        input_shape_vector, num_transformer_blocks, head_size, num_heads, ff_dim, dropout)

    point_feature = point_stream(inputs_point)
    vector_feature = vector_stream(inputs_vector)

    feature = keras.layers.concatenate([point_feature, vector_feature])
    x = feature
    # # conv
    # conv1 = keras.layers.Conv1D(
    #     filters=32, kernel_size=3, padding="same")(x)
    # conv1 = keras.layers.BatchNormalization()(conv1)
    # conv1 = keras.layers.ReLU()(conv1)
    # # conv2 = keras.layers.Conv1D(
    # #     filters=64, kernel_size=3, padding="same")(conv1)
    # # conv2 = keras.layers.BatchNormalization()(conv2)
    # # x = keras.layers.ReLU()(conv2)
    # x = layers.Dropout(0.25)(conv1)
    # ###########

    # for _ in range(num_transformer_blocks):
    #     x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # x = layers.GlobalAveragePooling1D(data_format="channels_first")(
    #     x)  # data_format="channels_first"
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model([inputs_point, inputs_vector], outputs)


"""## Train and evaluate"""

# input_shape = x_train.shape[1:]

# model = build_model(
#     input_shape,
#     head_size=256,
#     num_heads=4,
#     ff_dim=64,
#     num_transformer_blocks=4,
#     mlp_units=[128],
#     mlp_dropout=0.2,
#     dropout=0.25,
# )
model = build_model(
    input_shape_point=x_train_points.shape[1:
                                           ], input_shape_vector=x_train_vectors.shape[1:
                                                                                       ],
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss='sparse_categorical_crossentropy',
    # loss='poisson',
    # loss="mean_squared_error",

    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],  # "mae"
)
model.summary()

# callbacks = [keras.callbacks.ModelCheckpoint(
#     "Transformer_vector_best_model.h5", save_best_only=True, monitor="sparse_categorical_accuracy"
# ),
#     keras.callbacks.EarlyStopping(
#     patience=50, restore_best_weights=True)]

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "Transformer_vector_best_model.h5", save_best_only=True, monitor="sparse_categorical_accuracy"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="sparse_categorical_accuracy", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(
        monitor="sparse_categorical_accuracy", patience=50, verbose=1),
]

history = model.fit(
    [x_train_points, x_train_vectors],
    y_train_points,
    validation_split=0.2,
    epochs=500,
    batch_size=64,  # 64
    callbacks=callbacks,
)

# model.save('transformer_model.h5')
print("Training finish!")
model = keras.models.load_model("Transformer_vector_best_model.h5")

test_loss, test_acc = model.evaluate(
    [x_test_points, x_test_vectors], y_test_vectors, verbose=1)
print("Test Accuracy:", test_acc)
print("Test loss:", test_loss)

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

plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_" + "loss"])
plt.title("model loss")
plt.ylabel("loss", fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()
"""## Conclusions

In about 110-120 epochs (25s each on Colab), the model reaches a training
accuracy of ~0.95, validation accuracy of ~84 and a testing
accuracy of ~85, without hyperparameter tuning. And that is for a model
with less than 100k parameters. Of course, parameter count and accuracy could be
improved by a hyperparameter search and a more sophisticated learning rate
schedule, or a different optimizer.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/timeseries_transformer_classification) and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/timeseries_transformer_classification).
"""
# model_name = "Transformer_vector"

# model = keras.models.load_model(f"{model_name}_best_model.h5")
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
fig.savefig(f'Transformer_confusion_matrix.png')


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

img_array_vectors = x_test_vectors[1][tf.newaxis, ...]
img_array_points = x_test_points[1][tf.newaxis, ...]
img_array = [img_array_points, img_array_vectors]
heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, pred_index=0)
print(heatmap.shape)  # 19偵
plt.matshow(heatmap)
plt.show()
