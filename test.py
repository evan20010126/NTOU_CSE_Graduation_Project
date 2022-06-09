import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

model = tf.keras.models.load_model("Convolution_model.h5")
model.summary()
