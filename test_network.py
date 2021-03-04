from utils import config
from utils import utils
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
from utils.network import build_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image

utils.configure_gpu()

print("[INFO] loading test dataset...")
img = image.load_img("4_58.png",target_size=(224,224))
img = np.asarray(img).astype('float32')
plt.imshow(img)
img = np.expand_dims(img, axis=0)

print("[INFO] loading model...")
model = build_model(config.IMG_SHAPE)
opt = SGD(learning_rate=config.LEARNING_RATE)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"]
)
model.load_weights(config.MODEL_PATH)

preds = model.predict(img)
print(preds[0])
