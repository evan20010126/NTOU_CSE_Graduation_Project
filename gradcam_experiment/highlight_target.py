import cv2
import csv

path = r"./output_sample_videos/11.mp4"
filename = path.split('/')[-1]


def write_csv(file_name, all_data):
    print("Writing csv...")

    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(all_data)

    print("\Finish writing csv/")


key = 0
cap = cv2.VideoCapture(path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
counter = 1
frame_highlight_list = list()
while(True):
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break
    cv2.imshow('video', frame)
    print(f"frame: {counter}/{total}|fps: {fps}")
    counter += 1
    is_break = False
    while (True):
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            is_break = True
            break
        elif key == 88:
            frame_highlight_list.append(1)
            break
            # "X" 這偵比錯
        elif key == 86:
            frame_highlight_list.append(0)
            break
            # "V" 這偵比對
    if is_break:
        break
write_csv(
    f"gradcam_experiment/{filename.split('.')[0]}.csv", frame_highlight_list)
