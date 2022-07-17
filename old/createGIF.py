import os
import imageio


def create_gif(source, name, duration):
    frames = []
    for img in source:
        frames.append(imageio.imread(img))
    imageio.mimsave(name, frames, 'GIF', duration=duration)
    print("success")


path = os.chdir("../path_img/")
pic_list = os.listdir()
print(pic_list)
gif_name = f"test.gif"
duration_time = 0.05
create_gif(pic_list, gif_name, duration_time)
