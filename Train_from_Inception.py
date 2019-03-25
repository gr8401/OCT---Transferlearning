# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:18:04 2019

@author: Daniel_Ramsing
"""
from __future__ import print_function
import tensorflow as tf
import keras
from keras.applications.inception_v3 import InceptionV3#, conv2d_bn
from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import h5py
import matplotlib.pyplot as plt


def features_from_file(path, ctx):
    #open(os.path.join(direct, "5_1.txt"), "r")
    
    D_directory = 'D:\\OCT_Model_data\\'
    h5f = h5py.File(D_directory + path, 'r')
    batch_count = h5f['batches'].value
    print(ctx, 'batches:', batch_count)       
    
    def generator():
        while True:
            for batch_id in range(0, batch_count):
                X = h5f['features-' + str(batch_id)]
                y = h5f['labels-' + str(batch_id)]
                yield X, y
            
    return batch_count, generator()

train_steps_per_epoch, train_generator = features_from_file('train.h5', 'Train')
validation_steps, validation_data = features_from_file('validation.h5', 'Validation')


# Build model

inputs = Input(shape=(8, 8, 2048)) 
x = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
#x = conv2d(inputs, 64, 1, 1)   ## Opmærksom her
x = Dropout(0.5)(x) 
x = Flatten()(x) 
outputs = Dense(4, activation='softmax')(x) 
model = Model(inputs=inputs, outputs=outputs) 
model.compile(optimizer=optimizers.adam(lr=0.001), 
   loss='categorical_crossentropy', metrics=['acc'])
model.summary()


## Train model ##
# Setup a callback to save the best model
callbacks = [ 
    ModelCheckpoint('./output/model.features.{epoch:02d}-{val_acc:.2f}.hdf5', 
      monitor='val_acc', verbose=1, save_best_only=True, 
      mode='max', period=1),
             
    ## Reduce learning rate if it validation loss does not improve in 5 epochs, but never below min learning rate
    ReduceLROnPlateau(monitor='val_loss', verbose=1, 
     factor=0.5, patience=5, min_lr=0.00005)
            ]

with tf.device("/device:CPU:0"):
    history = model.fit_generator(
       generator=train_generator, 
       steps_per_epoch=train_steps_per_epoch,  
       validation_data=validation_data, 
       validation_steps=validation_steps,
       epochs=100, callbacks=callbacks)