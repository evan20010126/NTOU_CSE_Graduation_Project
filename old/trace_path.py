import numpy as np
import pandas as pd
import cv2

picture_path = "../path_img/"

sign_language_df = pd.read_csv("draw_img_test.csv", header=None)

hand_sequence = [(0, 1), (1, 2), (2, 3), (3, 4),
                 (0, 5), (5, 6), (6, 7), (7, 8),
                 (0, 9), (9, 10), (10, 11), (11, 12),
                 (0, 13), (13, 14), (14, 15), (15, 16),
                 (0, 17), (17, 18), (18, 19), (19, 20)]  # 20個向量

pose_sequence = [(11, 12),
                 (12, 14), (14, 16),
                 (11, 13), (13, 15), ]  # 7個向量->6個向量

point_number = len(hand_sequence*2) + len(pose_sequence)

colorlist_left = [(255, 150, 150), (255, 48, 48), (194, 0, 0), ]  # bgr
colorlist_pose = [(171, 255, 171), (13, 255, 13), (0, 178, 0), ]
colorlist_right = [(140, 140, 255), (38, 38, 255), (0, 0, 194), ]
thick = 3
circle_thickness = -1
radius = 1


def draw_path(df):
    data = df.to_numpy()
    length = len(data[0])//130
    counter = 1
    for x in range(length):  # 迭代每一偵
        path_img_w = 500
        path_img_h = 500
        large_scale = 100
        path_img_center = (250, 250)
        path_img = np.zeros((path_img_w, path_img_h, 3), np.uint8)
        path_img.fill(255)
        color_idx = 0
        for row in data:
            point_color_left = colorlist_left[color_idx % len(colorlist_left)]
            point_color_pose = colorlist_pose[color_idx % len(colorlist_pose)]
            point_color_right = colorlist_right[color_idx % len(
                colorlist_right)]
            color_idx += 1
            # row = data[0]
            # pose: 23個點 left/right:各21個點 23+21*2=65
            vector = row[1:]
            vector = vector.reshape(
                (vector.shape[0])//130, 65, 2)  # (幾偵, 點, xy)
            img = vector[x]
            pose_points = img[0:23]
            left_hand_points = img[23:23+21]
            right_hand_points = img[23+21:23+21+21]

            cv2.circle(path_img, (int(pose_points[0][0]*large_scale+path_img_center[0]), int(
                pose_points[0][1]*large_scale+path_img_center[0])), radius=radius, color=point_color_pose, thickness=circle_thickness)

            for p1, p2 in pose_sequence:
                point1 = (pose_points[p1] *
                          large_scale)+path_img_center[0]
                point2 = (pose_points[p2] *
                          large_scale)+path_img_center[1]
                cv2.line(path_img, (int(point1[0]), int(point1[1])),
                         (int(point2[0]),
                          int(point2[1])), point_color_pose, thick)
                cv2.circle(path_img, (int(point1[0]), int(
                    point1[1])), radius=1, color=point_color_pose, thickness=circle_thickness)
                cv2.circle(path_img, (int(point2[0]), int(
                    point2[1])), radius=1, color=point_color_pose, thickness=circle_thickness)

            for p1, p2 in hand_sequence:
                point1 = (left_hand_points[p1] *
                          large_scale)+path_img_center[0]
                point2 = (left_hand_points[p2] *
                          large_scale)+path_img_center[1]

                print(f"leftHand: {point1}")
                cv2.line(path_img, (int(point1[0]), int(point1[1])),
                         (int(point2[0]),
                          int(point2[1])), point_color_left, thick)
                cv2.circle(path_img, (int(point1[0]), int(
                    point1[1])), radius=radius, color=point_color_left, thickness=circle_thickness)
                cv2.circle(path_img, (int(point2[0]), int(
                    point2[1])), radius=radius, color=point_color_left, thickness=circle_thickness)

            for p1, p2 in hand_sequence:
                point1 = (right_hand_points[p1] *
                          large_scale)+path_img_center[0]
                point2 = (right_hand_points[p2] *
                          large_scale)+path_img_center[1]

                print(f"rightHand: {point1}")
                cv2.line(path_img, (int(point1[0]), int(point1[1])), (int(
                    point2[0]), int(point2[1])), point_color_right, thick)
                cv2.circle(path_img, (int(point1[0]), int(
                    point1[1])), radius=radius, color=point_color_right, thickness=circle_thickness)
                cv2.circle(path_img, (int(point2[0]), int(
                    point2[1])), radius=radius, color=point_color_right, thickness=circle_thickness)

        cv2.imshow('path img', path_img)
        counter_str = "%05d" % counter
        cv2.imwrite(f'{picture_path}{counter_str}.jpg', path_img)
        counter += 1
        # cv2.waitKey(0)

    cv2.destroyAllWindows()


draw_path(sign_language_df)
