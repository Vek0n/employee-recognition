# import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from utils import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



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
        horizontal_flip=True,
        brightness_range=[0.3,1.0], 
        zoom_range=0.2,
        rotation_range=20, 
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest'
    )
    
    train_data = trdata.flow_from_directory(directory="data/train",target_size=(224,224))
    tsdata = ImageDataGenerator()
    test_data = tsdata.flow_from_directory(directory="data/test", target_size=(224,224))
 
    return train_data, test_data


def configure_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = InteractiveSession(config=config)
    


def main():
    read_data()

if __name__ == "__main__":
    main()
