import cv2
import mediapipe as mp
import os
import openpyxl
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
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


all_keypoints = []


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
exist_leftHand = False
exist_rightHand = False


def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:

            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))

            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x,
                         hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [640, 480]).astype(int))

            output = text, coords

    return output


# <Main>
# 列出指定路徑底下所有檔案(包含資料夾)
allFileList = os.listdir(dirPath)  # allFileList: 為所有input影片檔案名稱

file_counter = 1
break_processing = False
for my_file in allFileList:
    if break_processing:
        break
    cap = cv2.VideoCapture(dirPath + '\\' + my_file)
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
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    # get_coordinates(hand_landmarks)
            print("hands:")
            print(results.multi_handedness)
            if (results.multi_handedness != None):
                break_processing = True
                break

            mp_drawing.draw_landmarks(
                image,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            print("pose:")
            print(results_pose.pose_landmarks)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break_processing = True
                break
    print(all_keypoints)
    cap.release()
