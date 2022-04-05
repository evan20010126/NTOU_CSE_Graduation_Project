# From Python
# It requires OpenCV installed for Python
import time
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import time
my_answer = list()
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
    # parser.add_argument('--save_video', type=bool, default=False,
    # help = 'To write output video. default name file_name.avi')
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    params["net_resolution"] = "320x176"
    params["hand_net_resolution"] = "256x256"
    params["hand"] = True
    params["hand_detector"] = 2
    params["body"] = 1
    params["video"] = "../../../examples/media/video.avi"
    # params["write_images"] = "hello"

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
    cap = cv2.VideoCapture("../../../examples/media/video.avi")
    while cap.isOpened():
        try:
            ret, imageToProcess = cap.read()

            # x, y = (185.692673, 303.112244)
            x, y = (185.692673 - 170, 303.112244 - 100)
            # dx = 157.587555
            dx = 157.587555 + 250
            # dy = 157.587555
            dy = 157.587555 + 250
            handRectangles = [
                # Left/Right hands person 0
                # [
                #     op.Rectangle(320.035889, 377.675049, 69.300949, 69.300949),
                #     op.Rectangle(0., 0., 0., 0.),
                # ],
                # Left/Right hands person 1
                # [
                #     op.Rectangle(80.155792, 407.673492, 80.812706, 80.812706),
                #     op.Rectangle(46.449715, 404.559753, 98.898178, 98.898178),
                # ],
                # Left/Right hands person 2
                [
                    op.Rectangle(x, y, dx, dy),
                    op.Rectangle(88.984360, 268.866547,
                                 117.818230, 117.818230),
                ]
            ]

            # Create new datum
            datum = op.Datum()
            datum.cvInputData = imageToProcess
            datum.handRectangles = handRectangles

            # Process and display image
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
            print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
            # print(type(datum.cvOutputData))
            # img = np.zeros((512, 512, 3), np.uint8)
            # print(type(img))
            img = cv2.rectangle(datum.cvOutputData, (int(x), int(y)),
                                (int(x + dx), int(y + dy)), (0, 255, 0), 1)

            # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", img)
            my_answer.append(img)
        except:
            break
    for im in my_answer:
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", im)
        # time.sleep(1)
        cv2.waitKey(50)
except Exception as e:
    print(e)
    sys.exit(-1)
