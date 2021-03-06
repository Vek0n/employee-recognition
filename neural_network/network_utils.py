import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from PIL import Image
import neural_network.network_config as config

def plot_training(H, plotPath):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)


def read_training_data():
    trdata = ImageDataGenerator(
        horizontal_flip = config.HORIZONTAL_FLIP,
        brightness_range = config.BRIGHTNESS_RANGE, 
        zoom_range = config.ZOOM_RANGE,
        rotation_range = config.ROTATION_RANGE, 
        shear_range = config.SHEAR_RANGE,
        width_shift_range = config.WIDTH_SHIFT_RANGE,
        height_shift_range = config.HEIGHT_SHIFT_RANGE,
        fill_mode = config.FILL_MODE 
    )
    train_data = trdata.flow_from_directory(
        directory=config.TRAINING_DATA_PATH, target_size = config.IMG_DIM
    )
    tsdata = ImageDataGenerator()
    test_data = tsdata.flow_from_directory(
        directory=config.TEST_DATA_PATH, target_size = config.IMG_DIM
    )
    return train_data, test_data


def configure_gpu():
    conf = ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.gpu_options.per_process_gpu_memory_fraction = config.PER_PROCESS_GPU_MEMORY_FRACTION
    session = InteractiveSession(config=conf)
