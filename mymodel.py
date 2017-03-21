from keras.models import Model
from keras.layers import Input,Dense,Dropout,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator 

def setupmodel():
    inputs = Input(shape = (1,48,48))
    x = ZeroPadding2D()(inputs)
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding2D()(x)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding2D()(x)
    x = Convolution2D(128, 3, 3, activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Convolution2D(128, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(6, activation='softmax')(x)

    model = Model(inputs,outputs)

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model
