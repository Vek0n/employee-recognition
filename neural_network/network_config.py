import os

IMG_SIZE = 224
IMG_DIM = (IMG_SIZE,IMG_SIZE)
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

#Training hyperparameters
BATCH_SIZE = 8
EPOCHS = 5
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 10
LEARNING_RATE = 0.0001
FEATURE_VECTOR_DIM = 4096
NUMBER_OF_CLASSES = 4
LOSS_FUNCTION = "categorical_crossentropy"
METRICS = "accuracy"

#Data augmentation
HORIZONTAL_FLIP = True
BRIGHTNESS_RANGE = [0.3,1.0]
ZOOM_RANGE = 0.2
ROTATION_RANGE = 20
SHEAR_RANGE = 0.2
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
FILL_MODE = 'nearest'

#Paths
PLOT_PATH = os.path.sep.join(["neural_network", "plots", "plot.png"])
MODEL_PATH = os.path.sep.join(["neural_network", "models", "model.h5"])
TRAINING_DATA_PATH = os.path.sep.join(["neural_network", "data", "train"])
TEST_DATA_PATH = os.path.sep.join(["neural_network", "data", "test"])

#Misc
PER_PROCESS_GPU_MEMORY_FRACTION = 0.8