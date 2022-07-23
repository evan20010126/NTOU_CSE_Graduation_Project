from tensorflow import keras
model = keras.models.load_model("Lstm_best_model.h5")
model.summary()
keras.utils.plot_model(model, show_shapes=True)
