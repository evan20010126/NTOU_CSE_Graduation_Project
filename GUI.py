
import tkinter as tk
import tkinter.messagebox
import cv2
import time
from functools import partial
import random
import tkinter.font as tkFont

from cv2 import sort
import mediapipe_webcam
import preprocess_userCSV
from tensorflow import keras
import numpy as np
import pandas as pd
import gradcam_detect
# def onOK():
#     # 取得輸入文字
#     print("Hello, {}確診.".format(entry.get()))
# # 輸入欄位
# entry = tk.Entry(menu,     # 輸入欄位所在視窗
#                  width=20)  # 輸入欄位的寬度
# entry.pack()

answer_classlist = ['Salty', 'Snack', 'Bubble Tea',
                    'Dumpling', 'Spicy', 'Sour', 'Sweet', 'Yummy']


menu = tk.Tk()
menu.title('Sign Language Interaction Tutorial System')
menu.geometry("800x500+250+150")  # window大小+左上角定位
menu['background'] = '#F4F1DE'

# -----Quiz start-----
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
        else:
            temp_row = np.append(temp_row, [-999.0, ])

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


def start_btn_func(target_class_num):
    # generate webcam.csv
    mediapipe_webcam.open_cam(SAVE_REC=False, SAVE_EXCEL=False,
                              SAVE_CSV=True, PREVIEW_INPUT_VIDEO_WITH_OPENPOSE_DETECT=True, cam_num=0)
    # generate webcam_stuff_zero.csv
    FTTAB = preprocess_userCSV.preprocess(max_column=27431)

    webcam_df = pd.read_csv("webcam_stuff_zero.csv",
                            header=None)
    x_test, y_test = split_target(webcam_df)
    del webcam_df
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    x_test = np.asarray(x_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)
    x_test = x_test.flatten().reshape(
        x_test.shape[0], (x_test.shape[1]//(point_number*2)), (point_number*2))
    # model = keras.models.load_model('Convolution_best_model.h5')
    select_model_name = 'Lstm_best_model.h5'
    model = keras.models.load_model(select_model_name)
    model.summary()
    predict_answer = model.predict(x_test)
    print("Predict: ", predict_answer)

    predict_answer = predict_answer.flatten().tolist()
    sort_predict_answer = sorted(predict_answer)

    # for i in range(len(sort_predict_answer)-1, len(sort_predict_answer)-3, -1):
    #     idx = predict_answer.index(sort_predict_answer[i])
    #     gradcam_detect.get_heapmap(model, -3, x_test[0], idx)
    idx = predict_answer.index(sort_predict_answer[-1])  # 判斷出的類別
    print(f"\033[92m{answer_classlist[idx]}")
    print("\033[0m")
    CORRECT = False

    if select_model_name == 'Transformer_best_model_vector.h5':
        layer_num = -5
    elif select_model_name == 'Transformer_best_model_points.h5':
        layer_num = -5
    else:
        layer_num = -3
    if (target_class_num == idx):
        # correct
        CORRECT = True
        second_idx = predict_answer.index(sort_predict_answer[-2])
        gradcam_detect.get_heapmap(
            model, layer_num, x_test[0], second_idx, FTTAB)
    else:
        # wrong
        CORRECT = False
        first_idx = predict_answer.index(sort_predict_answer[-1])
        gradcam_detect.get_heapmap(
            model, layer_num, x_test[0], first_idx, FTTAB)

    createScore(CORRECT)

# -----確認退出-----


def confirm_to_quit():
    if tk.messagebox.askokcancel('溫馨提示', '確定要退出嗎?'):
        menu.quit()

# -----開影片-----


def openVideo(num):
    v1 = cv2.VideoCapture(
        'C:\\Users\\yumi\\Desktop\\good\\smaplevideo\\snack\\bubbletea.avi')
    v2 = cv2.VideoCapture(
        'C:\\Users\\yumi\\Desktop\\good\\smaplevideo\\snack\\dumpling.avi')
    v3 = cv2.VideoCapture(
        'C:\\Users\\yumi\\Desktop\\good\\smaplevideo\\snack\\hot.avi')
    v4 = cv2.VideoCapture(
        r'..\media\salty\220511_222809-6237668.MOV')
    v5 = cv2.VideoCapture(
        'C:\\Users\\yumi\\Desktop\\good\\smaplevideo\\snack\\snack.avi')
    v6 = cv2.VideoCapture(
        'C:\\Users\\yumi\\Desktop\\good\\smaplevideo\\snack\\sour.avi')
    v7 = cv2.VideoCapture(
        'C:\\Users\\yumi\\Desktop\\good\\smaplevideo\\snack\\sweet.avi')
    v8 = cv2.VideoCapture(
        'C:\\Users\\yumi\\Desktop\\good\\smaplevideo\\snack\\taste.avi')
    if num == 1:
        cap = v1
    if num == 2:
        cap = v2
    if num == 3:
        cap = v3
    if num == 4:
        cap = v4
    if num == 5:
        cap = v5
    if num == 6:
        cap = v6
    if num == 7:
        cap = v7
    if num == 8:
        cap = v8

    # 以迴圈從影片檔案讀取影格，並顯示出來
    while(cap.isOpened()):
        ret, frame = cap.read()
        # time.sleep(0.05)
        cv2.imshow('frame', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----創TutorialVideo視窗-----


def createTutorialVideo():
    TutorialVideo = tk.Toplevel(menu)
    TutorialVideo.title('Tutorial Video')
    TutorialVideo.geometry("800x500")
    TutorialVideo['background'] = '#F4F1DE'

    btn1 = tk.Button(TutorialVideo, bg='#F2CC8F',
                     width=25, height=3, text='Bubble Tea', command=partial(openVideo, 1)).grid(row=0, column=0, padx=7, pady=35)
    btn2 = tk.Button(TutorialVideo, bg='#F2CC8F',
                     width=25, height=3, text='Dumpling', command=partial(openVideo, 2)).grid(row=0, column=1, padx=7, pady=35)
    btn3 = tk.Button(TutorialVideo, bg='#F2CC8F',
                     width=25, height=3, text='Hot', command=partial(openVideo, 3)).grid(row=0, column=2, padx=7, pady=35)
    btn4 = tk.Button(TutorialVideo, bg='#F2CC8F',
                     width=25, height=3, text='Salty', command=partial(openVideo, 4)).grid(row=0, column=3, padx=7, pady=35)
    btn5 = tk.Button(TutorialVideo, bg='#F2CC8F',
                     width=25, height=3, text='Snack', command=partial(openVideo, 5)).grid(row=1, column=0, padx=7, pady=35)
    btn6 = tk.Button(TutorialVideo, bg='#F2CC8F',
                     width=25, height=3, text='Sour', command=partial(openVideo, 6)).grid(row=1, column=1, padx=7, pady=35)
    btn7 = tk.Button(TutorialVideo, bg='#F2CC8F',
                     width=25, height=3, text='Sweet', command=partial(openVideo, 7)).grid(row=1, column=2, padx=7, pady=35)
    btn8 = tk.Button(TutorialVideo, bg='#F2CC8F',
                     width=25, height=3, text='Taste Good', command=partial(openVideo, 8)).grid(row=1, column=3, padx=7, pady=35)

# -----創Practice視窗-----


def createPractice():
    Practice = tk.Toplevel(menu)
    Practice.title('Practice')
    Practice.geometry("800x500")
    Practice['background'] = '#F4F1DE'

    btn1 = tk.Button(Practice, bg='#F2CC8F',
                     width=25, height=3,).grid(row=0, column=0, padx=7, pady=35)
    btn2 = tk.Button(Practice, bg='#F2CC8F',
                     width=25, height=3).grid(row=0, column=1, padx=7, pady=35)
    btn3 = tk.Button(Practice, bg='#F2CC8F',
                     width=25, height=3).grid(row=0, column=2, padx=7, pady=35)
    btn4 = tk.Button(Practice, bg='#F2CC8F',
                     width=25, height=3).grid(row=0, column=3, padx=7, pady=35)
    btn5 = tk.Button(Practice, bg='#F2CC8F',
                     width=25, height=3).grid(row=1, column=0, padx=7, pady=35)
    btn6 = tk.Button(Practice, bg='#F2CC8F',
                     width=25, height=3) .grid(row=1, column=1, padx=7, pady=35)
    btn7 = tk.Button(Practice, bg='#F2CC8F',
                     width=25, height=3).grid(row=1, column=2, padx=7, pady=35)
    btn8 = tk.Button(Practice, bg='#F2CC8F',
                     width=25, height=3).grid(row=1, column=3, padx=7, pady=35)

# -----創Quiz視窗-----


def createQuiz():
    Quiz = tk.Toplevel(menu)
    Quiz.title('Practice')
    Quiz.geometry("800x500")
    Quiz['background'] = '#F4F1DE'

    # 隨機標示文字
    # tt = random.choice(['Bubble Tea', 'Dumpling', 'Hot', 'Salty',
    #                     'Snack', 'Sour', 'Sweet', 'Taste Good'])
    answer_number = random.randint(0, 7)
    fontStyle = tkFont.Font(family="Lucida Grande", size=35)
    label = tk.Label(
        Quiz, text=answer_classlist[answer_number], font=fontStyle)
    label.place(relx=0.48, rely=0.25)
    start_btn = tk.Button(Quiz, text='Start', bg='#F2CC8F', width=40, command=partial(start_btn_func, answer_number),
                          height=3, cursor='star').place(relx=0.35, rely=0.5)

# -----創Score視窗-----


def createScore(CORRECT):
    Score = tk.Toplevel(menu)
    Score.title('My Score')
    Score.geometry("800x500")
    Score['background'] = '#F4F1DE'
    fontStyle = tkFont.Font(family="Lucida Grande", size=35)

    if CORRECT:
        label = tk.Label(
            Score, text="You are right!!", font=fontStyle)
        label.place(relx=0.48, rely=0.1)
    else:
        label = tk.Label(
            Score, text="跨謀", font=fontStyle)
        label.place(relx=0.48, rely=0.1)

    replay_btn = tk.Button(Score, text='Repaly', bg='#F2CC8F', width=40,
                           height=3, cursor='star').place(relx=0.35, rely=0.5)


# <Main>
# ------menu-----
Tutorial_btn = tk.Button(
    menu, text="Tutorial Videos", bg='#F2CC8F', width=600, height=3, cursor='heart', command=createTutorialVideo).pack(padx=30, pady=20)

Practice_btn = tk.Button(
    menu, text="Practice", bg='#F2CC8F', width=600, height=3, cursor='heart', command=createPractice).pack(padx=30, pady=20)

Quiz_btn = tk.Button(
    menu, text="Start A Quiz", bg='#F2CC8F', width=600, height=3, cursor='heart', command=createQuiz).pack(padx=30, pady=20)

quit_btn = tk.Button(
    menu, text='退出', command=confirm_to_quit).place(relx=0.02, rely=0.92)
menu.mainloop()
# ------menu-----


# class Tutorial_Videos():
# video_btn1 = tk.Button(
#     menu, text="Videos", bg='#F2CC8F', width=20, height=2, cursor='heart').pack(padx=30, pady=15)


# menu.mainloop()
