from random import randrange

from tkinter import Button
from turtle import color
import tensorflow as tf
from tensorflow import keras
import numpy as np
import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


important_frame = None


def get_heapmap(model, layer_num, testing_data, class_idx, FTTAB):
    global important_frame
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
            # FTTAB[i]可以對應到正確的偵數
            important_frame.append(FTTAB[i])
            # print(FTTAB[i])
    save_frames = list()
    # print(important_frame)
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
            # cv2.imwrite(
            #     f'output_sample_videos/webcam_important/wrongPose_img{ele}.jpg', frame)
            save_frames.append(frame)
            pre_ele = ele

    print('\033[0m')

    # plt.figure(figsize=(15, 8), dpi=100)
    # plt.figure()
    # plt.subplot()
    # plt.matshow(heatmap)
    # for i in range(0, len(save_frames)):
    #     plt.subplot(1, len(save_frames), i+1)
    #     # subplot: row, colunm, 操作的格子
    #     plt.imshow(save_frames[i][:, :, [2, 1, 0]])  # 設定圖片 img[高, 寬, channel]
    #     # 此資料結構: row-major, 若 img[0] 則輸出圖片的第一列
    #     plt.title(str(i))   # 設定title
    #     plt.axis('off')  # on: 顯示坐標軸; off: 不顯示座標軸

    # fig = plt.figure(figsize=(15, 8), dpi=100, )
    fig = plt.figure()
    rows_num = 7
    gs1 = gridspec.GridSpec(
        nrows=rows_num, ncols=len(save_frames), wspace=0.05)
    ax1 = fig.add_subplot(gs1[0, :])
    ax1.matshow(heatmap)

    for i in range(0, len(save_frames)):
        ax2 = fig.add_subplot(gs1[1:rows_num-1, i])
        ax2.imshow(save_frames[i][:, :, [2, 1, 0]])
        ax2.axis('off')  # on: 顯示坐標軸; off: 不顯示座標軸

    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # buttonaxe = plt.axes([0.94, 0.03, 0.03, 0.03])
    ax3 = fig.add_subplot(gs1[rows_num-1, :])
    # ax3.set_zorder(10)
    button1 = plt.Button(ax3, 'replay')
    button1.on_clicked(fn_maker(heatmap=heatmap, FTTAB=FTTAB))
    plt.subplots_adjust(left=0.03, right=0.98)
    plt.show()
    # heatmap = np.array([heatmap, heatmap, heatmap, heatmap, heatmap, heatmap])
    # heatmap = heatmap.reshape(6, -1)

    # 0820 modify
    # cv_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    # print(cv_img)
    # # cv_img = 1-cv_img
    # cv_img[:, :, 0] = 0
    # cv_img[:, :, 2] = 0
    # cv_img = cv2.resize(
    #     cv_img, (5*heatmap.shape[1], 20), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("sth", cv_img)


def get_heapmap_FOREACH(model, layer_num, testing_data, target_class_num, total_class_num, FTTAB, frame_cutting):
    print("target:" + str(target_class_num))
    global important_frame
    last_conv_layer_name = model.layers[layer_num].name
    # print(model.layers[-3].name)

    img_array_points = testing_data[0][tf.newaxis, ...]
    img_array_vectors = testing_data[1][tf.newaxis, ...]

    target_heatmap = None

    heatmaps = np.array(list())
    heatmaps_col_num = 0

    for cas in range(0, total_class_num):
        heatmap = make_gradcam_heatmap(
            [img_array_points, img_array_vectors], model, last_conv_layer_name, pred_index=cas)
        heatmap = heatmap.flatten()
        heatmaps_col_num = len(heatmap)
        if cas == target_class_num:
            target_heatmap = heatmap
        else:
            heatmaps = np.append(heatmaps, heatmap)

    heatmaps = heatmaps.reshape(-1, heatmaps_col_num)
    heatmap = np.array([])  # new heatmap (錯誤率)
    for frame_num in range(len(target_heatmap)):
        if (target_heatmap[frame_num] <= 0.5):
            col = heatmaps[:, frame_num]
            col = col[col >= 0.3]
            col_num = len(col)
            if col_num != 0:
                average = 1 - target_heatmap[frame_num]
            else:
                # this frame neural network think->not important
                average = 0.0  # 怎麼比都是對的
        else:
            average = 1 - target_heatmap[frame_num]
        heatmap = np.append(heatmap, average)

    print(heatmap.shape)  # 19偵
    heatmap = heatmap.reshape(-1, heatmap.shape[0])
    print(f"len(heatmap):{len(heatmap)}")
    print('\033[93m')
    important_frame = []
    # i = 0
    for i in range(0, heatmap.shape[-1]):
        # while (i < heatmap.shape[-1]):
        ele = heatmap[0][i]
        # print("heatmap")
        # print(heatmap)
        if (ele >= 0.5 and (FTTAB[i] not in important_frame)):
            # FTTAB[i]可以對應到正確的偵數
            important_frame.append(FTTAB[i])
            # print(FTTAB[i])
    save_frames = list()
    # print(important_frame)
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
            # cv2.imwrite(
            #     f'output_sample_videos/webcam_important/wrongPose_img{ele}.jpg', frame)
            save_frames.append(frame)
            pre_ele = ele

    print('\033[0m')

    fig = plt.figure()
    fig.suptitle(
        "Heatmap for error rate\n(yellow: wrong, purple: correct)")
    rows_num = 7
    if len(save_frames) == 0:
        gs1 = gridspec.GridSpec(
            nrows=rows_num, ncols=1, wspace=0.05)
    else:
        gs1 = gridspec.GridSpec(
            nrows=rows_num, ncols=len(save_frames), wspace=0.05)
    ax1 = fig.add_subplot(gs1[0, :])
    print(f"heatmap{heatmap}")
    # plt.matshow(heatmap)
    # plt.show()
    ax1.matshow(heatmap, vmax=1.0, vmin=0.0)

    for i in range(0, len(save_frames)):
        ax2 = fig.add_subplot(gs1[1:rows_num-1, i])
        ax2.imshow(save_frames[i][:, :, [2, 1, 0]])
        ax2.axis('off')  # on: 顯示坐標軸; off: 不顯示座標軸

    ax3 = fig.add_subplot(gs1[rows_num-1, :])
    button1 = plt.Button(ax3, 'replay')
    button1.on_clicked(
        fn_maker(heatmap=heatmap, FTTAB=FTTAB, frame_cutting=frame_cutting))
    plt.subplots_adjust(left=0.03, right=0.98)
    plt.show()


def fn_maker(heatmap, FTTAB, frame_cutting):
    color_list = [(175, 194, 118), (135, 207, 246),
                  (94, 153, 224), (92, 92, 199)]

    def review(event):
        try:
            important_frame = []
            # i = 0
            # 取平均 0 -> 0 -> 0 -> 1 -> 1 -> 1 -> 2 -> 2 -> 2
            score_list = []
            counter = 0
            for i in range(0, heatmap.shape[-1]):
                # FTTAB[i]可以對應到正確的偵數
                ele = heatmap[0][i]
                if (FTTAB[i] not in important_frame):
                    if (i != 0):
                        score_list[-1] /= counter  # 結算上一偵
                    counter = 1
                    important_frame.append(FTTAB[i])
                    score_list.append(ele)
                else:
                    score_list[-1] += ele
                    counter += 1
            score_list[-1] /= counter  # 最後一偵結算掉

            temp_score_list = []
            if frame_cutting > 1:
                for i in range(0, len(score_list)):
                    for j in range(frame_cutting):
                        temp_score_list.append(score_list[i])
                score_list = temp_score_list
            # ---依照heatmap數值填色---
            cap = cv2.VideoCapture('output_sample_videos/webcam.avi')

            print("-"*100)
            print(f"score list{score_list}")
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            print('height:{} width:{}'.format(height, width))
            # cv2.VideoCapture.get: Returns the specified VideoCapture property.
            # property 列表: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#:~:text=Enumerator-,CAP_PROP_POS_MSEC%C2%A0,Python%3A%20cv.CAP_PROP_READ_TIMEOUT_MSEC,-%E2%97%86%C2%A0

            frame_num = 0
            total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            def set_frame_number(x):
                nonlocal frame_num
                frame_num = x
                return

            click_close = 0

            def close_window(x):
                nonlocal click_close
                click_close = x
                return

            cv2.namedWindow('video file', flags=cv2.WINDOW_GUI_EXPANDED)
            # cv2.setWindowProperty(
            #     'video file', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.namedWindow('名稱')
            # 設定視窗的名稱

            # cv2.createTrackbar('----->Close', 'video file', 0, 1, close_window)

            cv2.createTrackbar('frame no.', 'video file', 0,
                               total_frame-1, set_frame_number)
            # 第一個參數時滑動條的名字，
            # 第二個參數是滑動條被放置的窗口的名字，
            # 第三個參數是滑動條默認值，
            # 第四個參數滑動條的最大值，
            # 第五個參數為 callback function, 當 trackbar 的值有改變時會觸發
            while frame_num < total_frame:
                # cv2.setTrackbarPos('frame no.', 'video file', frame_num)
                # cv2.setTrackbarPos() 設定 TrackbarPos 目前的位置
                # 第一個參數是滑動條名字，
                # 第二個時所在窗口，
                # 第三個參數是滑動條默認值，

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                # print(f'total_frame = {total_frame}; frame_num = {frame_num}')
                # 主要為此條指令配合 create 時的 callback function (設定frame_name) 達到 trackbar 拖曳時影片跟著動

                ret, frame = cap.read()
                if ret == False:
                    break
                if cv2.getWindowProperty('video file', 0) < 0:
                    break

                # 著色
                # print(type(frame))

                # 畫面著色version
                # frame[:, :, 2] = (
                #     255 * (score_list[frame_num])).astype(np.uint8)

                # 音量
                unit = width/200  # 5
                cv2.rectangle(frame, (int(width-unit*21), int(height-unit*101-unit*10)),
                              (int(width-unit*4), int(height-unit*9-unit*10)), (115, 93, 79), -1)
                for n in range(10):  # 畫框框
                    cv2.rectangle(frame, (int(width-unit*19), int(height-unit*19-unit*10-n*unit*9)),
                                  (int(width-unit*6), int(height-unit*11-unit*10-n*unit*9)), (0, 0, 0), 1)
                fill_num = int(score_list[frame_num]*10)
                for n in range(fill_num):  # 填顏色
                    if(n < 3):
                        this_color = color_list[0]
                    elif(n < 5):
                        this_color = color_list[1]
                    elif(n < 7):
                        this_color = color_list[2]
                    else:
                        this_color = color_list[3]
                    cv2.rectangle(frame, (int(width-unit*19), int(height-unit*19-unit*10-n*unit*9)),
                                  (int(width-unit*6), int(height-unit*11-unit*10-n*unit*9)), this_color, -1)

                cv2.rectangle(frame, (int(width-unit*60),
                                      int(height-unit*15)), (width, height), (115, 93, 79), -1)

                cv2.putText(frame, "Error Rate", (int(width-unit*58), int(height-unit*5)), cv2.FONT_HERSHEY_PLAIN,
                            2, color_list[0], 2, cv2.LINE_AA)
                # cv2.rectangle(frame, (125, 20), (175, 40), (255, 0, 255), -1)
                # cv2.rectangle(img, left_up, right_down, color, thickness)

                cv2.imshow('video file', frame)
                key = cv2.waitKey(20) & 0xFF
                # if key == 27:
                # if click_close == 1:
                # break
                # frame_num += 1

            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
    return review


def hello_its_me():
    cap = cv2.VideoCapture('output_sample_videos/webcam.avi')

    print('height:{} width:{}'.format(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
    # cv2.VideoCapture.get: Returns the specified VideoCapture property.
    # property 列表: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#:~:text=Enumerator-,CAP_PROP_POS_MSEC%C2%A0,Python%3A%20cv.CAP_PROP_READ_TIMEOUT_MSEC,-%E2%97%86%C2%A0

    frame_num = 0
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def set_frame_number(x):
        global frame_num
        frame_num = x
        return

    click_close = 0

    def close_window(x):
        global click_close
        click_close = x
        return

    cv2.namedWindow('video file')
    # cv2.namedWindow('名稱')
    # 設定視窗的名稱

    cv2.createTrackbar('OpenClose', 'video file', 0, 1, close_window)
    # cv2.createTrackbar('frame no.', 'video file', 0,
    #    total_frame-1, set_frame_number)
    # 第一個參數時滑動條的名字，
    # 第二個參數是滑動條被放置的窗口的名字，
    # 第三個參數是滑動條默認值，
    # 第四個參數滑動條的最大值，
    # 第五個參數為 callback function, 當 trackbar 的值有改變時會觸發

    while frame_num < total_frame:
        cv2.setTrackbarPos('OpenClose', 'video file', click_close)
        # cv2.setTrackbarPos('frame no.', 'video file', frame_num)
        # cv2.setTrackbarPos() 設定 TrackbarPos 目前的位置
        # 第一個參數是滑動條名字，
        # 第二個時所在窗口，
        # 第三個參數是滑動條默認值，

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        # 主要為此條指令配合 create 時的 callback function (設定frame_name) 達到 trackbar 拖曳時影片跟著動

        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imshow('video file', frame)
        key = cv2.waitKey(20) & 0xFF
        # if key == 27:
        if (click_close == 1):
            break
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()


# hello_its_me()
