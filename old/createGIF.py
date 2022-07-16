import os
import imageio
num = int(input())
def create_gif(source, name, duration):
    frames = []
    for img in source:
        frames.append(imageio.imread(img))
    imageio.mimsave(name, frames, 'GIF', duration = duration)
    print("處裡完成")
path = os.chdir(f".//fig{num}")
pic_list = os.listdir()
gif_name = f"result{num}.gif"
duration_time = 0.1
create_gif(pic_list, gif_name, duration_time)