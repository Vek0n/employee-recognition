from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from utils import config


def build_model(inputShape, embeddingDim=config.FEATURE_VECTOR_DIM, numberOfClasses=config.NUMBER_OF_CLASSES):
    
    inputs = Input(inputShape)

    x = Conv2D(filters=32, kernel_size=(20,20),padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(x)

    x = Conv2D(filters=64, kernel_size=(10,10),padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(x)

    x = Conv2D(filters=128, kernel_size=(5,5),padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters=256, kernel_size=(3,3),padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters=512, kernel_size=(3,3),padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(x)
    x = Dropout(0.3)(x)

    flattenedOutput= Flatten()(x)
    dense1 = Dense(units=embeddingDim, activation="relu")(flattenedOutput)
    dense2 = Dense(units=embeddingDim, activation="relu")(dense1)
    outputs = Dense(units=numberOfClasses, activation="softmax")(dense2)
 
    model = Model(inputs, outputs)

    return model