import os
import cv2

dirPath = r'C:\Users\User\Desktop\salty'  # Input video的資料夾路徑
allFileList = os.listdir(dirPath)  # allFileList: 為所有input影片檔案名稱

min = 999999
for my_file in allFileList:
    cap = cv2.VideoCapture(dirPath + '\\' + my_file)
    Total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if Total < min:
        min = Total
    print(min)
