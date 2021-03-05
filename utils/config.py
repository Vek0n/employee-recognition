import os

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

BATCH_SIZE = 8
EPOCHS = 5
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 10
LEARNING_RATE = 0.0001
FEATURE_VECTOR_DIM = 4096
NUMBER_OF_CLASSES = 4


PLOT_PATH = os.path.sep.join(["plots", "plot.png"])
MODEL_PATH = os.path.sep.join(["models", "model.h5"])

CAM_1_URL = "http://192.168.1.95:8081/video"