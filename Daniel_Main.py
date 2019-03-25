# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:42:34 2019

@author: Daniel_Ramsing
"""
from __future__ import print_function
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras import optimizers
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import h5py


conv_base = InceptionV3(weights='imagenet', include_top=False)


train_dir = r'C:\Users\Daniel Ramsing\Documents\GitHub\OCT---Transferlearning\images\CellData\OCT\train' 
validation_dir = r'C:\Users\Daniel Ramsing\Documents\GitHub\OCT---Transferlearning\images\CellData\OCT\test'



def extract_features(file_name, directory, key, 
   sample_count, target_size, batch_size, 
   class_mode='categorical'):
    
    D_directory = 'D:\\OCT_Model_data\\'
    h5_file = h5py.File(D_directory+file_name, 'w')
    datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(directory, 
      target_size=target_size,
      batch_size=batch_size, class_mode=class_mode)
    
    samples_processed = 0
    batch_number = 0
    batch_counter = 0
    if sample_count == 'all':
        sample_count = generator.n      
    print_size = True
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        
        if print_size == True:
            print_size = False
            print('Features shape', features_batch.shape)
            
        samples_processed += inputs_batch.shape[0]
        h5_file.create_dataset('features-'+ str(batch_number), data=features_batch)
        h5_file.create_dataset('labels-'+str(batch_number), data=labels_batch)
        batch_number = batch_number + 1
        batch_counter = batch_counter + 1
        if batch_counter == 25:
            print("Batch:%d Sample:%d\r" % (batch_number,samples_processed), end="")
            batch_counter = 0
        
        if samples_processed >= sample_count:
            break
  
    h5_file.create_dataset('batches', data=batch_number)
    h5_file.close()
    return



extract_features('train.h5', train_dir, 
   key='train', sample_count='all', 
   batch_size=100, target_size=(299,299))

extract_features('validation.h5', validation_dir,
  key='validation', sample_count='all', 
  batch_size=100, target_size=(299,299))