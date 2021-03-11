import neural_network.network_config as config
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from imutils.paths import list_images
from tensorflow.keras.preprocessing import image
from .network_utils import plot_training, read_training_data, configure_gpu
from .network_architecture import build_model

class NeuralNetwork:

    def __init__(self):
        np.set_printoptions(suppress=True)
        configure_gpu()
        K.clear_session()


    def get_model(self):
        model = build_model(config.IMG_SHAPE)
        model.load_weights(config.MODEL_PATH)
        return model


    def train_network(self):
        train_data, test_data = read_training_data()

        model = build_model(config.IMG_SHAPE)

        opt = SGD(learning_rate=config.LEARNING_RATE)
        model.compile(loss=config.LOSS_FUNCTION, optimizer=opt,
            metrics=[config.METRICS])

        history = model.fit(
            x=train_data,
            validation_data=test_data,
            epochs=config.EPOCHS
        )
        model.save_weights(config.MODEL_PATH) #TF == 2.0.0
        plot_training(history, config.PLOT_PATH)


    def get_predictions_for_image(self, image, model):
        img = np.asarray(image).astype('float32')
        img = np.expand_dims(img, axis=0)
        preds = model.predict(img)
        return preds[0]