
import tkinter as tk
import tkinter.messagebox
import cv2
import time
from functools import partial
import random
import tkinter.font as tkFont
import mediapipe_webcam

# def onOK():
#     # 取得輸入文字
#     print("Hello, {}確診.".format(entry.get()))
# # 輸入欄位
# entry = tk.Entry(menu,     # 輸入欄位所在視窗
#                  width=20)  # 輸入欄位的寬度
# entry.pack()


menu = tk.Tk()
menu.title('Sign Language Interaction Tutorial System')
menu.geometry("800x500+250+150")  # window大小+左上角定位
menu['background'] = '#F4F1DE'

# -----Webcam-----


def open_webcam():
    cap = cv2.VideoCapture(0)
    writer = cv2.VideoWriter('./samplevideo.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0,
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while(cap.isOpened()):
        ret, frame = cap.read()
        # time.sleep(0.05)
        cv2.imshow('frame', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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


def createQuiz():
    Quiz = tk.Toplevel(menu)
    Quiz.title('Practice')
    Quiz.geometry("800x500")
    Quiz['background'] = '#F4F1DE'

    # 隨機標示文字
    tt = random.choice(['Bubble Tea', 'Dumpling', 'Hot', 'Salty',
                        'Snack', 'Sour', 'Sweet', 'Taste Good'])
    fontStyle = tkFont.Font(family="Lucida Grande", size=35)
    label = tk.Label(Quiz, text=tt, font=fontStyle)
    label.place(relx=0.48, rely=0.25)
    star_btn = tk.Button(Quiz, text='Start', bg='#F2CC8F', width=40, command=partial(mediapipe_webcam.open_cam, SAVE_REC=False, SAVE_EXCEL=False,
                         SAVE_CSV=True, PREVIEW_INPUT_VIDEO_WITH_OPENPOSE_DETECT=True),
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
