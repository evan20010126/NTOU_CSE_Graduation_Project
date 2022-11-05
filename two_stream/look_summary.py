import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model(
    r"D:\openpose1\build\examples\NTOU_CSE_Graduation_Project\auto_leave_person\points_conv\-1\Convolution_best_model.h5")

model.summary()
