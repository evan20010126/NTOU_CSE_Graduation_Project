# 2022/03/22
# myself_webcam

import argparse
from sys import platform
import os
import sys
import cv2
import threading

# my_answer = list()
my_answer = dict()


def process_image(img, frame_num, opWrapper):
    x, y = (185.692673 - 170, 303.112244 - 100)
    dx = 157.587555 + 250
    dy = 157.587555 + 250
    handRectangles = [
        [
            op.Rectangle(x, y, dx, dy),
            op.Rectangle(88.984360, 268.866547,
                         117.818230, 117.818230),
        ]
    ]

    # Create new datum
    datum = op.Datum()
    datum.cvInputData = img
    datum.handRectangles = handRectangles

    # Process and display image
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    # print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    # print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))

    # print(type(datum.cvOutputData))
    # img = np.zeros((512, 512, 3), np.uint8)
    # print(type(img))
    img = cv2.rectangle(datum.cvOutputData, (int(x), int(y)),
                        (int(x + dx), int(y + dy)), (0, 255, 0), 1)
    my_answer[frame_num] = img
    cv2.imshow("hello", img)


def work():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", default="C:\\Users\\User\\Desktop\\openpose1\\examples\\media\\004.mp4")

    args = parser.parse_known_args()

    params = dict()
    params["model_folder"] = "../../../models/"
    params["net_resolution"] = "320x176"
    params["hand_net_resolution"] = "256x256"
    params["hand"] = True
    params["hand_detector"] = 2
    params["body"] = 1

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

    # Starting Openpose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process video
    # cap = cv2.VideoCapture(args[0].video)
    cap = cv2.VideoCapture(0)
    frame_num = -1
    threads = []

    while True:
        try:
            frame_num += 1
            ret, imageToProcess = cap.read()

            process_image(imageToProcess, frame_num, opWrapper)
            # threads.append(threading.Thread(target=process_image,
            #    args=(imageToProcess, frame_num, opWrapper)))
            # threads[frame_num].start()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.waitKey(50)
        except:
            break

    # 釋放攝影機
    cap.release()

    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()
    # for i in range(len(threads)):
    # threads[i].start()
    # for i in range(len(threads)):
    # threads[i].join()

    for key in my_answer:
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", my_answer[key])
        # time.sleep(1)
        cv2.waitKey(100)


# <Main>
try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            sys.path.append(dir_path + '/../../python/openpose/Release')
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + \
                '/../../x64/Release;' + dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            sys.path.append('../../python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    work()
except Exception as e:
    print(e)
    sys.exit(-1)
# </Main>
