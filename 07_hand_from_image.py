from multiprocessing.dummy import freeze_support
from sqlite3 import Time
import time
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import time
import multiprocessing as mp
my_answer = dict()
try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release')
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + \
                '/../../x64/Release;' + dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python')
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/img002.png",
                        help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    params["net_resolution"] = "320x176"
    params["hand_net_resolution"] = "256x256"
    params["hand"] = True
    params["hand_detector"] = 2
    params["face"] = True
    params["face_detector"] = 2
    params["body"] = 0
    params["video"] = "../../../examples/media/video.avi"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1:
            next_item = args[1][i+1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:
                params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params:
                params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read image and face rectangle locations
    video_source = "002.mp4"
    cap = cv2.VideoCapture("../../../examples/media/" + video_source)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("width: ", width)
    print("\nheight: ", height)
    # cap.set(3, width/8)
    # cap.set(4, height/8)
    counter = 0

    # scaling video function
    def rescale_frame(frame, percent):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    zoom_out = 0
    l1, l2, l3, l4 = 0, 0, 0, 0
    r1, r2, r3, r4 = 0, 0, 0, 0
    if(video_source == "001.mp4"):  # 問題(左手怪怪的)
        zoom_out = 25
        l1, l2, l3, l4 = (width/8)-170, (height/8)-30, 165, 165
        r1, r2, r3, r4 = (width/8), (height/8)-50, 200, 200
        f1, f2, f3, f4 = (width/8)-75, (height/8)-130, 150, 150
    elif(video_source == "002.mp4"):  # 喝(有點爛)
        zoom_out = 50
        l1, l2, l3, l4 = (width/4)-150, (height/4)-175, 300, 300
        r1, r2, r3, r4 = (width/4)-150, (height/4)-175, 300, 300
        f1, f2, f3, f4 = (width/4)-150, (height/4)-250, 300, 300
    elif(video_source == "003.mp4"):  # 東西
        zoom_out = 50
        l1, l2, l3, l4 = (width/4)-175, (height/4)-75, 325, 325
        r1, r2, r3, r4 = (width/4)-175, (height/4)-75, 325, 325
        f1, f2, f3, f4 = (width/4)-150, (height/4)-250, 300, 300
    elif(video_source == "004.mp4"):  # 酸
        zoom_out = 50
        l1, l2, l3, l4 = (width/4)-150, (height/4)-150, 300, 300
        r1, r2, r3, r4 = (width/4)-150, (height/4)-150, 300, 300
        f1, f2, f3, f4 = (width/4)-150, (height/4)-250, 300, 300
    elif(video_source == "005.mp4"):  # 大
        zoom_out = 25
        l1, l2, l3, l4 = (width/8)-155, (height/8)-120, 160, 160
        r1, r2, r3, r4 = (width/8)+15, (height/8)-120, 160, 160
        f1, f2, f3, f4 = (width/8)-50, (height/8)-150, 100, 100
    elif(video_source == "006.mp4"):  # 小
        zoom_out = 25
        l1, l2, l3, l4 = (width/8)-175, (height/8)-110, 180, 180
        r1, r2, r3, r4 = (width/8)+10, (height/8)-110, 170, 170
        f1, f2, f3, f4 = (width/8)-65, (height/8)-160, 130, 130

    handRectangles = [
        [
            op.Rectangle(l1, l2, l3, l4),
            op.Rectangle(r1, r2, r3, r4),
        ]
    ]

    faceRectangles = [
        op.Rectangle(f1, f2, f3, f4)
    ]

    def process_keypoint(imageToProcess, frame_num, opWrapper):
        imageToProcess = rescale_frame(imageToProcess, percent=zoom_out)

        # handRectangles = [
        #     [
        #         op.Rectangle(l1, l2, l3, l4),
        #         op.Rectangle(r1, r2, r3, r4),
        #     ]
        # ]

        # faceRectangles = [
        #     op.Rectangle(f1, f2, f3, f4)
        # ]

        # Create new datum
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        datum.handRectangles = handRectangles
        datum.faceRectangles = faceRectangles

        # Process and display image

        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
        # print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
        # print("Face keypoints: \n" + str(datum.faceKeypoints))
        # print(type(datum.cvOutputData))
        # img = np.zeros((512, 512, 3), np.uint8)
        # print(type(img))

        img = datum.cvOutputData
        cv2.rectangle(img, (int(l1), int(l2)),
                      (int(l1+l3), int(l2+l3)), (0, 255, 0), 2)
        cv2.rectangle(img, (int(r1), int(r2)),
                      (int(r1+r3), int(r2+r3)), (255, 0, 0), 2)
        cv2.rectangle(img, (int(f1), int(f2)),
                      (int(f1+f3), int(f2+f3)), (0, 255, 255), 2)
        # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", img)
        my_answer[frame_num] = img

    t1 = time.time()
    process_list = []
    frame_num = -1
    while cap.isOpened():
        counter += 1
        print("Enter while loop ", counter)
        frame_num += 1
        try:
            ret, imageToProcess = cap.read()
            process_list.append(mp.Process(target=process_keypoint,
                                           args=(imageToProcess, frame_num, opWrapper)))
        except Exception as e:
            print(e)
            break
        if(frame_num == 100):
            break
        # if(preimage in imageToProcess):
        #     break
        # preimage = imageToProcess
        # print(preimage)
    # if __name__ == '__main__':
    for i in range(len(process_list)):
        process_list[i].start()
    for i in range(len(process_list)):
        process_list[i].join()

    for i in range(len(my_answer)):
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", my_answer[i])
        # time.sleep(1)
        cv2.waitKey(80)

    t2 = time.time()
    print("time: ", t2-t1)

    cv2.waitKey(0)


except Exception as e:
    print(e)
    sys.exit(-1)

    # threead:  25.758331775665283
