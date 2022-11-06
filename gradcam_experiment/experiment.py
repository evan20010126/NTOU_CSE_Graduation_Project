import sys
import os
import pandas as pd
import numpy as np
import csv
from tensorflow import keras
abcdefg = True
sys.path.append(".")
# sys.path.append("..")
if abcdefg:
    import preprocess_userCSV
    import mediapipe_webcam
    import gradcam_detect


def save_f1summary(csv_data, video_file_name, input_type, select_model_name, signLanguageLabel, answer_frame, precision, recall, F1_score):
    csv_data.append(video_file_name)
    csv_data.append(input_type)
    csv_data.append(select_model_name.split('\\')[-1])
    csv_data.append(select_model_name.split('\\')[-2])
    csv_data.append(signLanguageLabel)
    csv_data.append(answer_frame.shape[0])
    csv_data.append(precision)
    csv_data.append(recall)
    csv_data.append(F1_score)

    # /*end*/
    preprocess_userCSV.write_csv(
        './gradcam_experiment/f1_Summary.csv', csv_data, 'a')


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


answer_classlist = ['Salty', 'Snack', 'Bubbletea',
                    'Dumpling', 'Spicy', 'Sour', 'Sweet', 'Yummy']

hand_sequence = [(0, 1), (1, 2), (2, 3), (3, 4),
                 (0, 5), (5, 6), (6, 7), (7, 8),
                 (0, 9), (9, 10), (10, 11), (11, 12),
                 (0, 13), (13, 14), (14, 15), (15, 16),
                 (0, 17), (17, 18), (18, 19), (19, 20)]  # 20個向量

pose_sequence = [(0, 12), (0, 11),
                 (12, 14), (14, 16),
                 (11, 13), (13, 15), ]  # 7個向量->6個向量

point_number = len(hand_sequence*2) + len(pose_sequence)

# /* Input START*/
models = [
    [
        r".\auto_leave_person\two_stream_encoder2\-1\Transformer_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\0\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\1\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\2\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\3\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\4\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\5\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\6\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\7\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\8\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\9\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\10\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\11\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\12\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\13\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\14\Lstm_best_model.h5",
        # r".\auto_leave_person\two_stream_lstm\15\Lstm_best_model.h5",
    ],  # two stream
    [
        # r".\auto_leave_person\points_lstm\-1\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\0\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\1\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\2\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\3\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\4\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\5\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\6\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\7\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\8\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\9\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\10\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\11\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\12\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\13\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\14\Lstm_best_model.h5",
        # r".\auto_leave_person\points_lstm\15\Lstm_best_model.h5",
    ],  # points
    [
        # r".\auto_leave_person\vector_lstm\-1\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\0\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\1\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\2\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\3\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\4\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\5\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\6\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\7\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\8\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\9\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\10\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\11\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\12\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\13\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\14\Lstm_best_model.h5",
        # r".\auto_leave_person\vector_lstm\15\Lstm_best_model.h5",

    ]  # vector
]

rootdirPath = r".\gradcam_experiment"
all_class_name = os.listdir(rootdirPath)
file_counter = 1


for label_name in all_class_name:
    if(label_name == "experiment.py" or label_name == "highlight_target.py" or label_name == "f1_Summary.csv"):
        continue
    signLanguageLabel = label_name
    target_class_num = answer_classlist.index(signLanguageLabel.capitalize())
    print("\033[92m")
    print(signLanguageLabel)
    print("\033[0m")
    file_counter = 1
    dirPath = f"{rootdirPath}\\{signLanguageLabel}"
    allFileList = os.listdir(dirPath)  # allFileList: 為所有input影片檔案名稱
    for video_file_name in allFileList:
        if (video_file_name.split('.')[-1] == "csv"):
            continue
        csv_filename = rootdirPath+"/" + \
            signLanguageLabel+"/" + video_file_name[0:-4] + ".csv"

        with open(csv_filename, newline='') as f:
            reader = csv.reader(f)
            answer_frame = np.array((list(reader)[0])).astype(np.float32)

        mediapipe_webcam.open_cam(SAVE_REC=False, SAVE_EXCEL=False, SAVE_CSV=True,
                                  PREVIEW_INPUT_VIDEO_WITH_OPENPOSE_DETECT=False, cam_num=(rootdirPath+"/" + signLanguageLabel+"/" + video_file_name))

        FTTAB, frame_cutting = preprocess_userCSV.preprocess(
            max_column=27301, src_csv_file='webcam.csv', dest_csv_file='webcam_stuff_zero.csv')

        # print(f"frame_cutting :{frame_cutting}")

        webcam_df_points = pd.read_csv("webcam_stuff_zero.csv",
                                       header=None)

        x_test_points, y_test_points = split_target_points(
            webcam_df_points)

        webcam_df_vectors = pd.read_csv("webcam_stuff_zero.csv",
                                        header=None)
        x_test_vectors, y_test_vectors = split_target(webcam_df_vectors)

        del webcam_df_points, webcam_df_vectors

        x_test_points = np.asarray(x_test_points).astype(np.float32)
        x_test_vectors = np.asarray(x_test_vectors).astype(np.float32)
        y_test_vectors = np.asarray(
            y_test_vectors).astype(np.float32)  # label一樣

        x_test_points = x_test_points.flatten().reshape(
            x_test_points.shape[0], x_test_points.shape[1]//130, 130)

        x_test_vectors = x_test_vectors.flatten().reshape(
            x_test_vectors.shape[0], (x_test_vectors.shape[1]//(point_number*2)), (point_number*2))

        # /* 以上 切割完成 */
        for select_model_name in models[0]:  # two stream
            if (select_model_name.split('\\')[-1][0] == "T"):
                layer_num = -5
            else:
                layer_num = -3

            model = keras.models.load_model(select_model_name)
            model.summary()

            score_list = gradcam_detect.get_heapmap_FOREACH(
                model, layer_num, [x_test_points[0], x_test_vectors[0]], target_class_num, len(answer_classlist), FTTAB, frame_cutting, VIEWER_GATE=False)  # score_list 是恢復後的影片

            score_list[score_list > 0.5] = 1.0  # "有比錯"
            score_list[score_list <= 0.5] = 0.0  # "沒錯"

            if answer_frame.shape[0] < score_list.shape[0]:
                score_list = np.resize(score_list, answer_frame.shape)
            # 0: 兩邊都說這偵沒錯 , 1:和正確答案不一樣, 2:兩邊都說他比錯
            result = (answer_frame + score_list)  # answer_frame & score_list

            # - precision - 命中/所有預測的結果
            if(np.count_nonzero(score_list == 1.0) == 0):
                precision = 1.0
            else:
                precision = np.count_nonzero(
                    result == 2.0) / np.count_nonzero(score_list == 1.0)

            # - recall - 命中 / 所有真正錯的
            if(np.count_nonzero(answer_frame == 1.0) == 0):
                recall = 1.0
            else:
                recall = np.count_nonzero(
                    result == 2.0) / np.count_nonzero(answer_frame == 1.0)
            # - F1_score - F1-score = 2 * Precision * Recall / (Precision + Recall)
            if (precision + recall) == 0:
                F1_score = 0
            else:
                F1_score = 2 * (precision * recall / (precision + recall))

            print(f"precision: {precision}")
            print(f"recall: {recall}")
            print(f"F1_score: {F1_score}")

            # /*save data*/
            csv_data = list()
            save_f1summary(csv_data, video_file_name, "two_stream", select_model_name,
                           signLanguageLabel, answer_frame, precision, recall, F1_score)
        for select_model_name in models[1]:  # point
            if (select_model_name.split('\\')[-1][0] == "T"):  # Transformer
                layer_num = -5
            else:
                layer_num = -3

            model = keras.models.load_model(select_model_name)
            model.summary()

            score_list = gradcam_detect.get_heapmap_FOREACH_oneChennel(
                model, layer_num, x_test_points[0], target_class_num, len(answer_classlist), FTTAB, frame_cutting, VIEWER_GATE=False)  # score_list 是恢復後的影片

            score_list[score_list > 0.5] = 1.0  # "有比錯"
            score_list[score_list <= 0.5] = 0.0  # "沒錯"

            if answer_frame.shape[0] < score_list.shape[0]:
                score_list = np.resize(score_list, answer_frame.shape)
            # 0: 兩邊都說這偵沒錯 , 1:和正確答案不一樣, 2:兩邊都說他比錯
            result = (answer_frame + score_list)  # answer_frame & score_list

            # - precision - 命中/所有預測的結果
            if(np.count_nonzero(score_list == 1.0) == 0):
                precision = 1.0
            else:
                precision = np.count_nonzero(
                    result == 2.0) / np.count_nonzero(score_list == 1.0)

            # - recall - 命中 / 所有真正錯的
            if(np.count_nonzero(answer_frame == 1.0) == 0):
                recall = 1.0
            else:
                recall = np.count_nonzero(
                    result == 2.0) / np.count_nonzero(answer_frame == 1.0)
            # - F1_score - F1-score = 2 * Precision * Recall / (Precision + Recall)
            if (precision + recall) == 0:
                F1_score = 0
            else:
                F1_score = 2 * (precision * recall / (precision + recall))

            print(f"precision: {precision}")
            print(f"recall: {recall}")
            print(f"F1_score: {F1_score}")

            # /*save data*/
            csv_data = list()
            save_f1summary(csv_data, video_file_name, "point", select_model_name,
                           signLanguageLabel, answer_frame, precision, recall, F1_score)
        for select_model_name in models[2]:  # vector
            if (select_model_name.split('\\')[-1][0] == "T"):  # Transformer
                layer_num = -5
            else:
                layer_num = -3

            model = keras.models.load_model(select_model_name)
            model.summary()

            score_list = gradcam_detect.get_heapmap_FOREACH_oneChennel(
                model, layer_num, x_test_vectors[0], target_class_num, len(answer_classlist), FTTAB, frame_cutting, VIEWER_GATE=False)  # score_list 是恢復後的影片

            score_list[score_list > 0.5] = 1.0  # "有比錯"
            score_list[score_list <= 0.5] = 0.0  # "沒錯"

            if answer_frame.shape[0] < score_list.shape[0]:
                score_list = np.resize(score_list, answer_frame.shape)
            # 0: 兩邊都說這偵沒錯 , 1:和正確答案不一樣, 2:兩邊都說他比錯
            result = (answer_frame + score_list)  # answer_frame & score_list

            # - precision - 命中/所有預測的結果
            if(np.count_nonzero(score_list == 1.0) == 0):
                precision = 1.0
            else:
                precision = np.count_nonzero(
                    result == 2.0) / np.count_nonzero(score_list == 1.0)

            # - recall - 命中 / 所有真正錯的
            if(np.count_nonzero(answer_frame == 1.0) == 0):
                recall = 1.0
            else:
                recall = np.count_nonzero(
                    result == 2.0) / np.count_nonzero(answer_frame == 1.0)
            # - F1_score - F1-score = 2 * Precision * Recall / (Precision + Recall)
            if (precision + recall) == 0:
                F1_score = 0
            else:
                F1_score = 2 * (precision * recall / (precision + recall))

            print(f"precision: {precision}")
            print(f"recall: {recall}")
            print(f"F1_score: {F1_score}")

            # /*save data*/
            csv_data = list()
            save_f1summary(csv_data, video_file_name, "vector", select_model_name,
                           signLanguageLabel, answer_frame, precision, recall, F1_score)
    # /* Input END */
