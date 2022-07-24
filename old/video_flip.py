import cv2
import numpy as np
import os

rootdirPath = r"..\media_flip"
all_class_name = os.listdir(rootdirPath)
file_counter = 1
break_processing = False

for label_name in all_class_name:
    signLanguageLabel = label_name
    file_counter = 1
    dirPath = f"{rootdirPath}\\{signLanguageLabel}"
    allFileList = os.listdir(dirPath)  # allFileList: 為所有input影片檔案名稱
    for my_file in allFileList:
        if break_processing:
            break
        cap = cv2.VideoCapture(f"{dirPath}\\{my_file}")
        # cap = cv2.VideoCapture("./py_practice/test.MOV")
        writer = cv2.VideoWriter(f'./output_flip_videos/{signLanguageLabel}/{my_file}', cv2.VideoWriter_fourcc(*'XVID'), 20.0,
                                 (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        file_counter += 1

        while(True):
            ret, frame = cap.read()  # 擷取一張影像
            # image = cv2.imread("lena.jpg")
            # cv2.imshow("before", frame)
            if ret == False:
                break

            frame2 = cv2.flip(frame, 1)  # 左右水平翻轉

            writer.write(frame2)
            # cv2.imshow('capture', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break_processing = True
                break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    # cv2.imshow("after", frame2)
    # cv2.waitKey(0)
