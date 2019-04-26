# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:55:04 2019

@author: danie

Preprocess pipeline for a single image, mainly used for
development, testing and benchmarking
"""


import os
import PreProcess_Util as ppu
from skimage import io
import skimage.morphology as mp
import skimage.measure as ms
import numpy as np


import matplotlib.pyplot as plt

# List of images for trials
# NORMAL, DRUSEN, DME eller CNV NORMAL-1001666-1 #NORMAL-1001772-1 #DME-1805818-2
# DRUSEN-1047803-4
# DME-306172-20
# DRUSEN.JPG
# CNV-6666538-176  Langt billede   # CNV-6666538-421
# NORMAL-1016042-34      # NORMAL-1016042-8      # NORMAL-1016042-2

# input path
path = 'C:\\Users\\danie\\Desktop\\Misc\\ST7\\OCT---Transferlearning\\'
filename = os.path.join(path, 'NORMAL-1114127-8.JPEG')  # join with image path

# read image and normalise, uint8 = 2^8-1 = 255
test_im = io.imread(filename)
test_im = test_im/(2**(8)-1)


def main_test(test_im, threshold):  
    # Prep image by removal of artefacts in bottom and top, threshold and remove artefact columns
    # Output = (input_image, threshold, # of columns to remove at edge of picture)
    dog = ppu.prep_im(test_im, threshold, 5)
    
    # Structural element disk shaped size 3
    selem = mp.disk(3)
    
    # Dilation operation
    dog_dilate = mp.dilation(dog, selem);
    
    # Label regions (connected component analysis)
    dog_label = ms.label(dog_dilate)
    
    # first part of region comparison
    plt.figure(1);plt.subplot(1,2,1);plt.imshow(dog)
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
    
    # second part of plot for region comparison
    plt.subplot(1,2,2);plt.imshow(dog);
    
    
    # Normal polynomial approach, not used in this implementation
    # except for comparison
    x_test, y_test_unweighted = ppu.norm_poly_app(dog, 0.01)
    
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

    # Actually produce polynomial function
    poly = np.poly1d(bestfit)
    
    # make x-coordinates = length of image
    x = np.linspace(0, test_im.shape[1]-1, test_im.shape[1])
    
    # get y-coordinates from poly func
    y = poly(x)
    
    # Display #
    # Lots of specifics meant for saving, can be left out
    fig1 = plt.figure(2, figsize = (496*1.3/139, 512*1.3/139), dpi = 139);
    
    plt.axis('off');
    plt.imshow(test_im);
    plt.plot(x_test, y_test_unweighted, color="green", marker='o', markersize =2, label="Unweighted");
    plt.plot(x, y, color="blue", marker='o', markersize =2, label="RANSAC")
    return dog, x, y

threshold = -0.00018
dog, x, y = main_test(test_im, threshold)


# How many pixels do we hit of our Bruch's membrane region
hitpixels = ppu.hitpixels(dog, x, y)
hitpixels_perc = hitpixels/test_im.shape[1]
# Lower threshold if amount of hitpixels was unsastisfactory - slow
'''
for i in range(1,4):
    if hitpixels_perc < 0.5:
        dog, x, y = main_test(test_im, threshold+0.00002*i)
        hitpixels_perc = ppu.hitpixels(dog, x, y)/test_im.shape[1]
    else:
        break
'''
if hitpixels_perc > 0.4:
    # Roll image
    rolled_im, n_roll_test = ppu.roll_place_im(y, test_im)
'''
    # Crop, currently permanently crops to 512x496, not very robust!
    crop_roll_im = ppu.crop(rolled_im)
    # Save image with same name, but add -preprocess.
    # define good_path
    ppu.save(files[f1],crop_roll_im, good_path)
'''
# plot
plt.figure(3);plt.subplot(1,2,1);plt.imshow(test_im);
plt.subplot(1,2,2);plt.imshow(rolled_im)




#print(np.size(np.where(dog ==1)))