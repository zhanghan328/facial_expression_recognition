#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:56:03 2017

@author: wind
"""

from keras.models import Model
from keras.layers import Input,Dense,Dropout,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator 
import mymodel

model = mymodel.setupmodel()
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical')

test_datagen = test_datagen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical')

callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0)]
model.fit_generator(
        train_generator,
        samples_per_epoch=28273,
        nb_epoch=500,
        callbacks=callbacks,
        validation_data=test_datagen,
        nb_val_samples=800)
json_string = model.to_json()  
open('model.json','w').write(json_string)  
model.save_weights('model.h5')

