import os

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

BATCH_SIZE = 8
EPOCHS = 2
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 10
LEARNING_RATE = 0.0001
FEATURE_VECTOR_DIM = 4096
NUMBER_OF_CLASSES = 4


PLOT_PATH = os.path.sep.join(["plots", "plot.png"])
MODEL_PATH = os.path.sep.join(["models", "model.h5"])