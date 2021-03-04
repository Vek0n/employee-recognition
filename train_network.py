from utils.network import build_model
from utils import config
from utils import utils
from tensorflow.keras.optimizers import SGD
import numpy as np
from tensorflow.keras import backend as K

utils.configure_gpu()

print("[INFO] clearing session...")
K.clear_session()

print("[INFO] loading dataset...")
train_data, test_data = utils.read_data()

print("[INFO] building network...")
model = build_model(config.IMG_SHAPE)

print("[INFO] compiling model...")
opt = SGD(learning_rate=config.LEARNING_RATE)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

print("[INFO] training model...")
history = model.fit_generator(
    generator=train_data,
    validation_data=test_data,
    steps_per_epoch=config.STEPS_PER_EPOCH,
    validation_steps=config.VALIDATION_STEPS,
    epochs=config.EPOCHS
)

print("[INFO] saving  model...")
model.save_weights(config.MODEL_PATH) #TF == 2.0.0

print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)