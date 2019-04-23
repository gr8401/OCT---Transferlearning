# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:55:04 2019

@author: danie
"""


import os
import PreProcess_Util as ppu
from skimage import io
from skimage.filters import gaussian
import skimage.morphology as mp
import skimage.measure as ms
import numpy as np
from sklearn.metrics import confusion_matrix
import scipy.signal as sc
from tqdm import tqdm

import matplotlib.pyplot as plt

# NORMAL, DRUSEN, DME eller CNV NORMAL-1001666-1 #NORMAL-1001772-1 #DME-1805818-2
# DRUSEN-1047803-4
# DME-306172-20
# DRUSEN.JPG
# CNV-6666538-176   # CNV-6666538-421

'''
path = 'C:\\Users\\danie\\Desktop\\Misc\\ST7\\OCT---Transferlearning\\'
filename = os.path.join(path, 'CNV-6666538-421.JPEG')
#filename = 'C:\\Users\\danie\\Desktop\\ST8\\Projekt\\Data\\NORMAL-1001666-1.jpg'
test_im = io.imread(filename)
test_im = test_im/(2**(8)-1) # normaliserer
'''
#test_med = sc.medfilt(test_im, 7)

def main_test(test_im, threshold):
    test_im2 = test_im >0.9
    for pred_region in ms.regionprops(ms.label(test_im2)):
        minr, minc, maxr, maxc = pred_region.bbox
        if maxr >len(test_im)-1 or minr == 0:
            # Finder koordinater for regionen
            Coord = pred_region.coords
            Coord1 = Coord[:, 0]
            Coord2 = Coord[:, 1]
            test_im[Coord1, Coord2] = 0
    
    # Forbered billedet med threshold og søjle fjernelse
    # Output = (input_image, threshold, # of columns to remove at edge of picture)
    dog = ppu.prep_im(test_im, threshold, 5)
    
    # Strukturelt element i diskform med stoerrelse 3
    selem = mp.disk(3)
    
    # Dilation operation
    dog_dilate = mp.dilation(dog, selem);
    
    # Label regioner
    dog_label = ms.label(dog_dilate)
    
    '''
    For hver region, find start raekke og soejle og vurdér om regionen er (1) stoerre
    end 2000 pixels, 
    ##### (2) og (3) er ikke længere med da centroideremoval var smartere #####
    (2) om den har pixel vaerdier indenfor de foerste 100 raekker
    og (3) om der er pixel vaerdier indenfor de sidste 100 raekker. 
    #####                                                                 #####
    Er et af disse opfyldt, sae slet regionen i det filtrerede billede
    ### NYT ### Nu slettes der ogsae i regions billedet med henblik paa en sekundaer fjerning af billeder senere
    '''
    #plt.figure(1);plt.subplot(1,2,1);plt.imshow(dog)
    
    for pred_region in ms.regionprops(dog_label):
        #minr, minc, maxr, maxc = pred_region.bbox
        if pred_region.area < 2000:
             # Finder koordinater for regionen
             Coord = pred_region.coords
             Coord1 = Coord[:, 0]
             Coord2 = Coord[:, 1]
             # Fjerner nu baede regionerne i thresholded billede OG regionsbilleder
             # Nu uden forloekke
             dog[Coord1, Coord2] = 0
             dog_label[Coord1, Coord2] = 0
    
    # Laver en regions variabel NOTE! Kan ogsaa bruges til at goere ovenstaende 
    # for loop mere forstaelig
    pred_region = ms.regionprops(dog_label)
    
    
    # Fjerner regioner baseret paa centroide
    # Input er labellede regioner og tilsvarende det billede, regioner ønskes fjernet i.
    dog = ppu.remove_based_centroid(pred_region, dog)
    
    
    #plt.subplot(1,2,2);plt.imshow(dog);
    
    
    # Normal polynomie tilgang
    # Input er billede, hvortil man vil have fittet parabel og alpha parameter, som afgoer hvor fleksibel kurven er
    # output er x og y- koordinat vektorer
    
    x_test, y_test_unweighted = ppu.norm_poly_app(dog, 0.01)
    
    ##!!!!!!!!! ALTERNATIV Metode - RANSAC !!!!!!!!!!!!!!##
    
    dog_zeros = np.nonzero(dog)
    
    '''
    Saenker iterativt noedvendige antal punkter i parabelfittet indtil vi har et bestfit
    '''
    for i in reversed(range(50)):
        bestfit = ppu.ransac_polyfit(dog_zeros[1], dog_zeros[0], n = i)
        if bestfit is not None:
            #check = i
            break
    if bestfit is None:
        NoneAlarm = True
        x = 1
        y = 1
    else:
        poly = np.poly1d(bestfit)
        NoneAlarm = False
        x = np.linspace(0, test_im.shape[1]-1, test_im.shape[1])
        y = poly(x)
    
    # Display
    # Specificerer stoerrelse til print, kan undvaeres
    '''
    fig1 = plt.figure(2, figsize = (496*1.3/139, 512*1.3/139), dpi = 139);
    plt.axis('off');
    plt.imshow(test_im);
    plt.plot(x_test, y_test_unweighted, color="green", marker='o', markersize =2, label="Unweighted");
    plt.plot(x, y, color="blue", marker='o', markersize =2, label="RANSAC")
    '''
    return dog, x, y, NoneAlarm

img_dir = 'C:\\Users\\danie\\Desktop\\ST8\\Projekt\\Data\\OCT2017\\SaveTest\\'
data, files = ppu.load(img_dir)


for f1 in tqdm(range(len(data))):
    data[f1] = data[f1]/(2**(8)-1)
    threshold = -0.00018
    dog, x, y, NoneAlarm = main_test(data[f1], threshold)
    if NoneAlarm:
        ppu.save(files[f1], data[f1], 'Bad_pre')
        continue
    
    # Finder hvor mange pixels af vores polynomie rammer den paagaeldende region
    hitpixels = ppu.hitpixels(dog, x, y)
    hitpixels_perc = hitpixels/dog.shape[1]
    '''
    # Indsaet tjek om antal hitpixels og hvis ikke, forsoeg igen?
    for i in range(1,4):
        if hitpixels_perc < 0.5:
            dog, x, y = main_test(data[f1], threshold+0.00002*i)
            hitpixels = ppu.hitpixels(dog, x, y)
        else:
            break
    '''    
    if hitpixels_perc > 0.5:
        # Ruller billedet, rullet_billede = roll_im(y_koordinater, inputbillede)
        rolled_im, n_roll_test = ppu.roll_place_im(y, data[f1])
        crop_roll_im = ppu.crop(rolled_im)
        ppu.save(files[f1],crop_roll_im, 'Good_pre')
    else:
        #rolled_im, n_roll_test = ppu.roll_place_im(y, data[f1])
        ppu.save(files[f1], data[f1], 'Bad_pre')
    

# Gem billede
'''
plt.figure(3);plt.subplot(1,2,1);plt.imshow(test_im);
plt.subplot(1,2,2);plt.imshow(rolled_im)
'''



#print(np.size(np.where(dog ==1)))