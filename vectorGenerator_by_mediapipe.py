import cv2
import mediapipe as mp
import os
import openpyxl
import numpy as np

#-------------------------------------------------------------#
# Switch
SAVE_REC = False  # 是否將有姿態辨識過後的影片存檔在output_sample_videos
SAVE_EXCEL = False  # 是否儲存特徵點到output.xlsx
PREVIEW_INPUT_VIDEO_WITH_OPENPOSE_DETECT = True  # 是否預覽帶有姿態辨識過後的完整(無裁切)影片
#-------------------------------------------------------------#
# Input argument
signLanguageLabel = "salty"  # 鹹:salty 小吃:snack
# Input video的資料夾路徑
dirPath = r'..\media\salty'
#-------------------------------------------------------------#

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
all_keypoints = list()
previous_hand = ""

# 將特徵點存入excel


def write_xlsx(file_name, all_data):
    global signLanguageLabel
    # wksheet = wkbook["工作表1"]
    try:
        wkbook = openpyxl.load_workbook(file_name)
        # wkbook = openpyxl.Workbook(write_only=True)

    except Exception as e:
        # print(e)
        # 新增空白excel
        fn = file_name
        wkbook = openpyxl.Workbook()
        wkbook.save(fn)

    try:
        wksheet = wkbook["Sheet1"]
    except Exception as e:
        # print(e)
        # 新增空白sheet
        wksheet = wkbook.create_sheet("Sheet1", 0)
        wkbook.save(fn)

    wksheet.cell(row=1, column=1).value = "condition"
    begining = wksheet.max_row + 1
    i = 1

    wksheet.cell(row=begining, column=i).value = signLanguageLabel
    i += 1
    for data in all_data:
        try:
            wksheet.cell(row=begining, column=i).value = data[0]
            wksheet.cell(row=begining, column=i + 1).value = data[1]
            i = i + 2
        except Exception as e:
            print("資料格式錯誤")

    # wksheet.append(data.tolist())
    # for data in all_data:
    #     wksheet.append(data[0])
    #     wksheet.append(data[1])
    wkbook.save(file_name)

    # 關檔
    wb = openpyxl.Workbook(write_only=True)
    wb.close()

# get landmarks


# def get_coordinates(Landmark, mode):
#     """
#     Get bounding box coordinates for a hand landmark.
#     Args:
#         handLadmark: A HandLandmark object.
#         image_shape: A tuple of the form (height, width).
#     Returns:
#         A tuple of the form (xmin, ymin, xmax, ymax).
#     """
#  # store all x and y points in list
#     if mode == 0:
#         for i in range(21):
#             # multiply x by image width (原本有*image_shape[1])
#             all_keypoints.append(Landmark.landmark[i].x)
#             # multiply y by image height (原本有*image_shape[0])
#             all_keypoints.append(Landmark.landmark[i].y)
#     if mode == 1:
#         for i in range(21):
#             # multiply x by image width (原本有*image_shape[1])
#             all_keypoints.append(Landmark.landmark[i].x)
#             # multiply y by image height (原本有*image_shape[0])
#             all_keypoints.append(Landmark.landmark[i].y)
#     return
first = True
miss_point = False


def get_label_and_points(index, hand, results):
    global previous_hand, first
    global all_keypoints
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:

            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            # text = '{} {}'.format(label, round(score, 2))

            # Extract Coordinates
            # coords = tuple(np.multiply(
            #     np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x,
            #              hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            #     [640, 480]).astype(int))
            # print("label: ", label)
            if first:
                first = False
                previous_hand = label
                if label == "Left":
                    for i in range(21):
                        all_keypoints.append(hand.landmark[i].x)
                        all_keypoints.append(hand.landmark[i].y)
                elif label == "Right":
                    for i in range(42):
                        all_keypoints.append(0)
                    for i in range(21):
                        all_keypoints.append(hand.landmark[i].x)
                        all_keypoints.append(hand.landmark[i].y)
            else:
                if previous_hand == label:
                    for i in range(42):
                        print("add 0")
                        all_keypoints.append(0)
                    for i in range(21):
                        all_keypoints.append(hand.landmark[i].x)
                        all_keypoints.append(hand.landmark[i].y)
                else:
                    for i in range(21):
                        all_keypoints.append(hand.landmark[i].x)
                        all_keypoints.append(hand.landmark[i].y)

                previous_hand = label
        # print(all_keypoints)
        # print("hand.landmark.size: ", len(hand.landmark)) # = 21

    return


# <Main>
# 列出指定路徑底下所有檔案(包含資料夾)
allFileList = os.listdir(dirPath)  # allFileList: 為所有input影片檔案名稱

file_counter = 1
break_processing = False
for my_file in allFileList:
    if break_processing:
        break
    cap = cv2.VideoCapture(f"{dirPath}\\{my_file}")
    print(f"video: {file_counter} / {len(allFileList)}")
    file_counter += 1
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands, mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            results_pose = pose.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                image,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # print("pose:")
            if results_pose.pose_landmarks:  # 當有偵測到pose
                for i in range(23):  # 上半身的點(0~22)
                    print(f"pose{i}")
                    print(results_pose.pose_landmarks.landmark[i])
                    if (results_pose.pose_landmarks.landmark[i].visibility >= 0.5):
                        # ! 先隨便設，信心度超過0.5才填
                        all_keypoints.append(
                            results_pose.pose_landmarks.landmark[i].x)
                        all_keypoints.append(
                            results_pose.pose_landmarks.landmark[i].y)
                    else:
                        #! 信心度太低就填0
                        all_keypoints.append(0)
                        all_keypoints.append(0)
                # print(results_pose.pose_landmarks.landmark[0].x)
            else:
                for i in range(46):  # (23*2) = 46
                    all_keypoints.append(0)

            if results.multi_hand_landmarks:
                # num代表有抓到幾隻手
                for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    get_label_and_points(num, hand_landmarks, results)
                    # print(all_keypoints)
            else:
                for i in range(84):
                    all_keypoints.append(0)
            # print("hands:")
            # print(results.multi_handedness)
            # if (results.multi_handedness != None):
            #     break_processing = True
            #     break

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break_processing = True
                break
    cap.release()

print(all_keypoints)

f = open("te", mode="w")
f.write(all_keypoints.__str__())
f.close()
