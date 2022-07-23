from random import randrange
import tensorflow as tf
from tensorflow import keras
import numpy as np
import numpy
import matplotlib.pyplot as plt
import cv2


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


def get_heapmap(model, layer_num, testing_data, class_idx, FTTAB):

    last_conv_layer_name = model.layers[layer_num].name
    # print(model.layers[-3].name)

    img_array = testing_data[tf.newaxis, ...]

    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, pred_index=class_idx)
    print(heatmap.shape)  # 19偵
    heatmap = heatmap.reshape(-1, heatmap.shape[0])
    print(f"len(heatmap):{len(heatmap)}")
    print('\033[93m')
    important_frame = []
    # i = 0
    for i in range(0, heatmap.shape[-1]):
        # while (i < heatmap.shape[-1]):
        ele = heatmap[0][i]
        if (ele >= 0.3 and (FTTAB[i] not in important_frame)):
            important_frame.append(FTTAB[i])
            print(FTTAB[i])

    print(important_frame)
    cap = cv2.VideoCapture("output_sample_videos/webcam.avi")
    pre_ele = -2e9
    for ele in important_frame:
        # ret, frame = cap.read()
        # if(frame_idx in important_frame):
        #     cv2.imwrite(f"wrongPose_img{}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, ele)
        # 主要為此條指令配合 create 時的 callback function (設定frame_name) 達到 trackbar 拖曳時影片跟著動
        ret, frame = cap.read()
        if ret == False:
            break
        if (ele-pre_ele) >= 5:
            cv2.imwrite(
                f'output_sample_videos/webcam_important/wrongPose_img{ele}.jpg', frame)
        pre_ele = ele

    print('\033[0m')
    plt.matshow(heatmap)
    plt.show()
