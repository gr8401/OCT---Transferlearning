# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:55:04 2019

@author: danie

Preprocess.py but ofr multiple images, it is a bit slow and could still be optimized
"""


import os

import PreProcess_Util as ppu

from skimage import io
import skimage.morphology as mp
import skimage.measure as ms

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt


def main_test(test_im, threshold):
   
    # Prep image by removal of artefacts in bottom and top, threshold and remove artefact columns
    # Output = (input_image, threshold, # of columns to remove at edge of picture)
    dog = ppu.prep_im(test_im, threshold, 5)
    
    # Structural element disk shaped size 3
    selem = mp.disk(3)
    
    # Dilation operation
    dog_dilate = mp.dilation(dog, selem);
    
    # Label regions
    dog_label = ms.label(dog_dilate)
    
    '''
    For every region, remove any region with area less than 2000 pixels
    '''
    for pred_region in ms.regionprops(dog_label):
        # following lines can be used for more criterias involving region
        # positions:
        #minr, minc, maxr, maxc = pred_region.bbox
        if pred_region.area < 2000:
             # Coordinates for a given region
             Coord = pred_region.coords
             Coord1 = Coord[:, 0]
             Coord2 = Coord[:, 1]
             # Remove region in both thresholded image and labelled image
             dog[Coord1, Coord2] = 0
             dog_label[Coord1, Coord2] = 0
    
   # Region variable for inspection purposes
    pred_region = ms.regionprops(dog_label)
    
    
    # Remove regions based on centroid location and vertical position
    # Input is region properties and image with regions for removal
    dog = ppu.remove_based_centroid(pred_region, dog)
    
    # Normal polynomial approach, not used in this implementation
    #x_test, y_test_unweighted = ppu.norm_poly_app(dog, 0.01)
    
    ##!!!!!!!!! ALTERNATIV Method - RANSAC !!!!!!!!!!!!!!##
    
    # Get region coordinates (all other image coordinates are zero, due to thresholding)
    dog_zeros = np.nonzero(dog)
    
    '''
    Iteratively lowers necessary points for a good polynomial fit
    Allows for fitting of harder polynomial fits
    '''
    for i in reversed(range(50)):
        bestfit = ppu.ransac_polyfit(dog_zeros[1], dog_zeros[0], n = i)
        if bestfit is not None:
            #check = i
            break
    # If no polynomial, raise flag and input dummy values for x and y
    if bestfit is None:
        NoneAlarm = True
        x = 1
        y = 1
    else:
        # fit polynomial, don't raise flag, create x-vector and get y-coordinates
        poly = np.poly1d(bestfit)
        NoneAlarm = False
        x = np.linspace(0, test_im.shape[1]-1, test_im.shape[1])
        y = poly(x)
        
    return dog, x, y, NoneAlarm

# Directory setup
img_dir = 'C:\\Users\\danie\\Desktop\\ST8\\Projekt\\Data\\OCT2017\\train\\NORMAL4\\'
good_path = 'NORMAL4_Good'
bad_path = 'NORMAL4_Bad'

# Load images, slow, recommended to batch these into <10000 images a piece
data, files = ppu.load(img_dir)

# Genoptag fra fejl, her skal fejlnr gerne vaere et par numre mindre end aktuelt
'''
fejlnr = 9406
data = data[fejlnr:len(data)]
files = files[fejlnr:len(files)]
'''

# For every image in data, normalise, and run maintest
for f1 in tqdm(range(len(data))):
    data[f1] = data[f1]/(2**(8)-1)
    threshold = -0.00018
    dog, x, y, NoneAlarm = main_test(data[f1], threshold)
    # If no polynomial found, save as bad image
    if NoneAlarm:
        ppu.save(files[f1], data[f1], bad_path)
        continue
    
    # How many pixels do we hit of our Bruch's membrane region
    hitpixels = ppu.hitpixels(dog, x, y)
    hitpixels_perc = hitpixels/dog.shape[1]
    # Lower threshold if amount of hitpixels was unsastisfactory - slow
    '''
    # Indsaet tjek om antal hitpixels og hvis ikke, forsoeg igen?
    for i in range(1,4):
        if hitpixels_perc < 0.5:
            dog, x, y = main_test(data[f1], threshold+0.00002*i)
            hitpixels = ppu.hitpixels(dog, x, y)
        else:
            break
    '''    
    if hitpixels_perc > 0.4:
        # Roll image, roll_im(y_coordinates, inputimg)
        rolled_im, n_roll_test = ppu.roll_place_im(y, data[f1])
        crop_roll_im = ppu.crop(rolled_im)
        ppu.save(files[f1],crop_roll_im, good_path)
    else:
        #rolled_im, n_roll_test = ppu.roll_place_im(y, data[f1])
        ppu.save(files[f1], data[f1], bad_path)
    
