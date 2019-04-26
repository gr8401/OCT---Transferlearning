# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:10:11 2019

@author: Daniel_Ramsing

Inspired by: 
    RISE: Randomized Input Sampling for Explanation of Black-box Models
    arXiv:1806.07421v3
    
And using implementation details from:
    https://github.com/eclique/RISE
    https://github.com/shivshankar20/eyediseases-AI-keras-imagenet-inception
"""

import os
import numpy as np

import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from keras.models import load_model
from keras import backend as K

from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors

import requests

from tqdm import tqdm

from skimage.transform import resize

##################### Global variables: #####################
# Define classes for easier interpretation
classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Path to image(s)
path = r'C:\Users\danie\Desktop\Misc\ST7\OCT---Transferlearning\DRUSEN-1047803-4-preprocessed.jpeg'


# load base model --> Inceptionv3
base_model = InceptionV3(weights='imagenet', include_top=False)

# Load our model (load_model is a keras function imported at top)
our_model = load_model('./output/4conv_64_1_pool_09-0.94.hdf5')

# img target size
img_target_size = (512,496)

# How many masks to test each batch
batch_size = 100

# Mask variables:
N = 1000            # number of masks to generate
s = 8               # who knows
p1 = 0.5            # who knows

##################### Define own functions #####################
# Prepare image by loading, casting to array and correcting axes
def prep_img(path):
    # Load image, also uses inbuilt Keras functions, see imports
    img = load_img(path, target_size = img_target_size)
    # Normalise in order to get class probabilities, not just 1 or 0 for binary classification
    x = img_to_array(img)/255
    x = np.expand_dims(x, axis = 0)
    return img, x 

# Predict, first basemodel then our model, used in explain function and to generate a normal prediction without heatmap
def predict_img(img_as_array):
    x = base_model.predict(img_as_array)
    pred =our_model.predict(x)
    
    return pred
    
# Generate random masks to obscure parts of input image
def generate_masks(N, s, p1):
    cell_size = np.ceil(np.array(img_target_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *img_target_size))

    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + img_target_size[0], y:y + img_target_size[1]]
    masks = masks.reshape(-1, *img_target_size, 1)
    return masks
    

'''
Explain function, input:
x = image as array
masks = generated masks

Multiplies masks and images, then predicts in batches and finally produces a saliency map for each class,
through linear algebra (dot product)
'''
def explain(x, masks):
    preds = []
    # If not working, check if multiplying correct axes!
    masked = x * masks
    
    for i in tqdm(range(0, N, batch_size), desc = 'Predicting on masked images...'):
        # add masked predictions to list in batches of 'batch_size' up to total masks
        preds.append(predict_img(masked[i:min(i+batch_size, N)]))
    
    # organize all predicted masks in matrix
    preds = np.concatenate(preds)
    # produce one saliency map by dot product between masks
    sal_map = preds.T.dot(masks.reshape(N,-1)).reshape(-1,*img_target_size)
    sal_map = sal_map/N/p1
    
    return sal_map
        
        
##################### Main matter #####################


# Keras backend --> do not train, only predict
K.set_learning_phase(0)


img, x = prep_img(path)

masks = generate_masks(N, s, p1)

sal_map = explain(x, masks)

Pred_prob = predict_img(x)
Pred_prob = Pred_prob.reshape(len(classes))
class_idx = np.argmax(Pred_prob)


print('Image prediction was:')
print(classes[class_idx] + '\n')
print('with probability:')
print(Pred_prob[class_idx])


plt.figure
plt.axis('off')
plt.imshow(img)
plt.imshow(sal_map[class_idx], cmap = 'jet', alpha = 0.5)
plt.show()
