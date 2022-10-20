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
from PIL import Image, ImageTk, ImageSequence
import time

# def onOK():
#     # 取得輸入文字
#     print("Hello, {}確診.".format(entry.get()))
# # 輸入欄位
# entry = tk.Entry(menu,     # 輸入欄位所在視窗
#                  width=20)  # 輸入欄位的寬度
# entry.pack()

answer_classlist = ['Salty', 'Snack', 'Bubble_Tea',
                    'Dumpling', 'Spicy', 'Sour', 'Sweet', 'Yummy']


menu = tk.Tk()
menu.title('Sign Language Interaction Tutorial System')
menu['background'] = '#FAF2E9'

w = 800  # width for the Tk root
h = 500  # height for the Tk root
ws = menu.winfo_screenwidth()  # width of the screen
hs = menu.winfo_screenheight()  # height of the screen
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)
menu.geometry('%dx%d+%d+%d' % (w, h, x, y))
# menu.geometry("800x500+250+150")  # window大小+左上角定位
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


def split_target_points(new_data_df):
    new_data = new_data_df.to_numpy()
    y = new_data[:, 0]
    x = new_data[:, 1:]

    y[y == "salty"] = 0.0
    y[y == "snack"] = 1.0
    y[y == "bubbletea"] = 2.0
    y[y == "dumpling"] = 3.0
    y[y == "spicy"] = 4.0
    y[y == "sour"] = 5.0
    y[y == "sweet"] = 6.0
    y[y == "yummy"] = 7.0
    y[y == "webcam"] = -999.0

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
    FTTAB, frame_cutting = preprocess_userCSV.preprocess(max_column=27301)

    webcam_df_points = pd.read_csv("webcam_stuff_zero.csv",
                                   header=None)
    x_test_points, y_test_points = split_target_points(webcam_df_points)

    webcam_df_vectors = pd.read_csv("webcam_stuff_zero.csv",
                                    header=None)
    x_test_vectors, y_test_vectors = split_target(webcam_df_vectors)
    del webcam_df_points, webcam_df_vectors

    x_test_points = np.asarray(x_test_points).astype(np.float32)
    x_test_vectors = np.asarray(x_test_vectors).astype(np.float32)
    y_test_vectors = np.asarray(y_test_vectors).astype(np.float32)

    x_test_points = x_test_points.flatten().reshape(
        x_test_points.shape[0], x_test_points.shape[1]//130, 130)

    x_test_vectors = x_test_vectors.flatten().reshape(
        x_test_vectors.shape[0], (x_test_vectors.shape[1]//(point_number*2)), (point_number*2))
    #!change model
    # model = keras.models.load_model('Convolution_best_model.h5')
    select_model_name = 'Convolution_best_model.h5'
    model = keras.models.load_model(select_model_name)
    model.summary()
    predict_answer = model.predict([x_test_points, x_test_vectors])
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

    if select_model_name == 'Transformer_best_model.h5':
        layer_num = -5
    elif select_model_name == 'Transformer_best_model.h5':
        layer_num = -5
    else:
        layer_num = -3
    if (target_class_num == idx):
        # correct
        CORRECT = True
        # second_idx = predict_answer.index(sort_predict_answer[-2])
        # gradcam_detect.get_heapmap(
        #     model, layer_num, x_test[0], second_idx, FTTAB)
        gradcam_detect.get_heapmap_FOREACH(
            model, layer_num, [x_test_points[0], x_test_vectors[0]], target_class_num, len(answer_classlist), FTTAB, frame_cutting)
    else:
        # wrong
        CORRECT = False
        # first_idx = predict_answer.index(sort_predict_answer[-1])
        # gradcam_detect.get_heapmap(
        #     model, layer_num, x_test[0], first_idx, FTTAB)
        gradcam_detect.get_heapmap_FOREACH(
            model, layer_num, [x_test_points[0], x_test_vectors[0]], target_class_num, len(answer_classlist), FTTAB, frame_cutting)

    createScore(CORRECT, idx)

# -----確認退出-----


def confirm_to_quit(page):
    if page == menu:
        if tk.messagebox.askokcancel('溫馨提示', '確定要退出嗎?'):
            # page.quit()
            page.destroy()
    else:
        page.destroy()


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
    global w, h, x, y
    TutorialVideo.geometry('%dx%d+%d+%d' % (w, h, x, y))
    # TutorialVideo.geometry("800x500")
    TutorialVideo['background'] = '#FAF2E9'

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
    global w, h, x, y
    global answer_classlist
    Practice = tk.Toplevel(menu)
    Practice.title('Practice')
    Practice.geometry('%dx%d+%d+%d' % (w, h, x, y))
    Practice.geometry("800x500")
    Practice['background'] = '#FAF2E9'

    btn1 = tk.Button(Practice, text=answer_classlist[0], bg='#F2CC8F',
                     width=25, height=3,  command=partial(start_btn_func, 0)).grid(row=0, column=0, padx=7, pady=35)
    btn2 = tk.Button(Practice, text=answer_classlist[1], bg='#F2CC8F',
                     width=25, height=3, command=partial(start_btn_func, 1)).grid(row=0, column=1, padx=7, pady=35)
    btn3 = tk.Button(Practice, text=answer_classlist[2], bg='#F2CC8F',
                     width=25, height=3, command=partial(start_btn_func, 2)).grid(row=0, column=2, padx=7, pady=35)
    btn4 = tk.Button(Practice, text=answer_classlist[3], bg='#F2CC8F',
                     width=25, height=3, command=partial(start_btn_func, 3)).grid(row=0, column=3, padx=7, pady=35)
    btn5 = tk.Button(Practice, text=answer_classlist[4], bg='#F2CC8F',
                     width=25, height=3, command=partial(start_btn_func, 4)).grid(row=1, column=0, padx=7, pady=35)
    btn6 = tk.Button(Practice, text=answer_classlist[5], bg='#F2CC8F',
                     width=25, height=3, command=partial(start_btn_func, 5)).grid(row=1, column=1, padx=7, pady=35)
    btn7 = tk.Button(Practice, text=answer_classlist[6], bg='#F2CC8F',
                     width=25, height=3, command=partial(start_btn_func, 6)).grid(row=1, column=2, padx=7, pady=35)
    btn8 = tk.Button(Practice, text=answer_classlist[7], bg='#F2CC8F',
                     width=25, height=3, command=partial(start_btn_func, 7)).grid(row=1, column=3, padx=7, pady=35)

# -----創Quiz視窗-----


def createQuiz():
    global w, h, x, y
    Quiz = tk.Toplevel(menu)
    Quiz.title('Quiz')
    Quiz.geometry('%dx%d+%d+%d' % (w, h, x, y))
    # Quiz.geometry("800x500")
    Quiz['background'] = '#FAF2E9'

    # 隨機標示文字
    # tt = random.choice(['Bubble Tea', 'Dumpling', 'Hot', 'Salty',
    #                     'Snack', 'Sour', 'Sweet', 'Taste Good'])
    global topic_img
    global topic
    answer_number = random.randint(0, 7)
    question_heading = answer_classlist[answer_number]
    topic_img = Image.open((f'GUI_img\{question_heading}_topic.png'))
    # -----------
    # topic_img = Image.open(('GUI_img\Bubble_Tea_topic.png'))
    topic_img = topic_img.resize((300, 150))
    topic = ImageTk.PhotoImage(topic_img)
    canvas = tk.Canvas(Quiz, width=300, height=300, bg="#FAF2E9")
    # 在 Canvas 中放入圖片
    canvas.create_image(150, 150, anchor='center', image=topic)
    # ------------
    canvas.pack()
    # fontStyle = tkFont.Font(family="Lucida Grande", size=35)
    # label = tk.Label(
    #     Quiz, text=answer_classlist[answer_number], font=fontStyle)
    # label.place(relx=0.4, rely=0.5)
    start_btn = tk.Button(Quiz, text='Start', bg='#F2CC8F', width=40, command=partial(start_btn_func, answer_number),
                          height=5, cursor='star').place(relx=0.33, rely=0.6)
    # -----------
    # img = []
    # global a, flag
    # while 1:
    #     im = Image.open('GUI_img\please-begging.gif')
    #     im = im.resize((300, 300))
    #     # GIF图片流的迭代器
    #     iter = ImageSequence.Iterator(im)
    #     # frame就是gif的每一帧，转换一下格式就能显示了
    #     for frame in iter:
    #         pic = ImageTk.PhotoImage(frame)
    #         # pic = pic.resize((100, 100))
    #         canvas.create_image((0, 0), image=pic)
    #         time.sleep(0.1)
    #         Quiz.update_idletasks()  # 刷新
    #         Quiz.update()

    #     quit_btn = tk.Button(
    #         Quiz, text='上一頁', command=partial(confirm_to_quit, Quiz)).place(relx=0.02, rely=0.92)
    # ---------------
    # quit_btn = tk.Button(
    #     Quiz, text='上一頁', command=partial(confirm_to_quit, Quiz)).place(relx=0.02, rely=0.92)

# -----GIF-----


# -----創Score視窗-----
def createScore(CORRECT, idx):
    global w, h, x, y
    global answer_classlist

    Score = tk.Toplevel(menu)
    Score.title('My Score')
    Score.geometry('%dx%d+%d+%d' % (w, h, x, y))
    # Score.geometry("800x500")
    Score['background'] = '#FAF2E9'
    fontStyle = tkFont.Font(family="Lucida Grande", size=35)

    img = []

    if CORRECT:
        # label = tk.Label(
        #     Score, text="You are right!!", font=fontStyle)
        # label.place(relx=0.45, rely=0.15)
        # 開圖片
        global answer_classlist
        img333 = Image.open(('GUI_img\Bingo.png'))
        img333 = img333.resize((400, 200))
        global tk_img333
        tk_img333 = ImageTk.PhotoImage(img333)
        cute_pic_name = answer_classlist[idx]
        img_cute = Image.open((f'GUI_img\cute_{cute_pic_name}.png'))
        img_cute = img_cute.resize((200, 100))
        global tea
        tea = ImageTk.PhotoImage(img_cute)
        canvas = tk.Canvas(Score, width=400, height=400, bg="#FAF2E9")
        # 在 Canvas 中放入圖片
        canvas.create_image(200, 100, anchor='center', image=tk_img333)
        canvas.create_image(200, 250, anchor='center', image=tea)
        canvas.pack()
        # -----------
        # global a, flag
        # while 1:
        #     im = Image.open('GUI_img\giphy.gif')
        #     # GIF图片流的迭代器
        #     iter = ImageSequence.Iterator(im)
        #     # frame就是gif的每一帧，转换一下格式就能显示了
        #     for frame in iter:
        #         pic = ImageTk.PhotoImage(frame)
        #         pic = pic.resize((100, 100))
        #         canvas.create_image((100, 200), image=pic)
        #         time.sleep(0.1)
        #         Score.update_idletasks()  # 刷新
        #         Score.update()

        #     quit_btn = tk.Button(
        #         Score, text='上一頁', command=partial(confirm_to_quit, Score)).place(relx=0.02, rely=0.92)
        # ---------------
      # # correct_img = tk.PhotoImage(file='GUI_img\correct_img.gif')
      # label_img = tk.Label(Score, image=correct_img)
      # label_img.pack()
    else:
        # label = tk.Label(
        #     Score, text=f"跨謀\n我猜這是{answer_classlist[idx]}", font=fontStyle)
        # label.place(relx=0.45, rely=0.15)
        unknow_img = Image.open(('GUI_img\wrong_answer.png'))
        unknow_img = unknow_img.resize((400, 250))
        global tk_unknow_img
        tk_unknow_img = ImageTk.PhotoImage(unknow_img)
        heading = answer_classlist[idx]
        unknown_topic_img = Image.open((f'GUI_img\{heading}_topic.png'))
        unknown_topic_img = unknown_topic_img.resize((200, 100))
        global tk_unknown_topic_img
        tk_unknown_topic_img = ImageTk.PhotoImage(unknown_topic_img)
        canvas = tk.Canvas(Score, width=400, height=400, bg="#FAF2E9")
        # 在 Canvas 中放入圖片
        canvas.create_image(200, 150, anchor='center', image=tk_unknow_img)
        canvas.create_image(200, 300, anchor='center',
                            image=tk_unknown_topic_img)
        canvas.pack()

    # Gif_canvas = tk.Canvas(Score, width=300, height=300, bg='red')

    # while 1:
    #     im = Image.open('GUI_img\crying_cat.gif')
    #     # GIF图片流的迭代器
    #     iter = ImageSequence.Iterator(im)
    #     # frame就是gif的每一帧，转换一下格式就能显示了
    #     for frame in iter:
    #         pic = ImageTk.PhotoImage(frame)
    #         Gif_canvas.create_image((300, 300), image=pic)
    #         time.sleep(0.1)
    #         Score.update_idletasks()  # 刷新
    #         Score.update()

    #     quit_btn = tk.Button(
    #         Score, text='上一頁', command=partial(confirm_to_quit, Score)).place(relx=0.02, rely=0.92)
    replay_btn = tk.Button(Score, text='Repaly', bg='#F2CC8F', width=15,
                           height=3, cursor='star').place(relx=0.82, rely=0.85)
    quit_btn = tk.Button(
        Score, text='上一頁', command=partial(confirm_to_quit, Score)).place(relx=0.02, rely=0.92)


# <Main>
# ------menu-----
Tutorial_btn = tk.Button(
    menu, text="Tutorial Videos", bg='#F2CC8F', width=600, height=3, cursor='heart', command=createTutorialVideo).pack(padx=30, pady=20)

Practice_btn = tk.Button(
    menu, text="Practice", bg='#F2CC8F', width=600, height=3, cursor='heart', command=createPractice).pack(padx=30, pady=20)

Quiz_btn = tk.Button(
    menu, text="Start A Quiz", bg='#F2CC8F', width=600, height=3, cursor='heart', command=createQuiz).pack(padx=30, pady=20)

quit_btn = tk.Button(
    menu, text='退出', command=partial(confirm_to_quit, menu)).place(relx=0.02, rely=0.92)
menu.mainloop()
# ------menu-----


# class Tutorial_Videos():
# video_btn1 = tk.Button(
#     menu, text="Videos", bg='#F2CC8F', width=20, height=2, cursor='heart').pack(padx=30, pady=15)


# menu.mainloop()
