# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:55:04 2019

@author: danie
"""

import os, os.path
import PreProcess_Util as ppu
from skimage import io
from skimage.filters import gaussian
import skimage.morphology as mp
import skimage.measure as ms
import numpy as np
from sklearn.metrics import confusion_matrix
import scipy.signal as sc

import matplotlib.pyplot as plt

#TINA (NÆSTE 5 LINJER):
import cv2
import glob
from os import listdir
from PIL import Image as PImage
from skimage import io as skimageio

def feature(x, order=2):
    """Generate polynomial feature of the form
    [1, x, x^2, ..., x^order] where x is the column of x-coordinates
    and 1 is the column of ones for the intercept.
    """
    x = x.reshape(-1, 1)
    return np.power(x, np.arange(order+1).reshape(1, -1)) 

# NORMAL, DRUSEN, DME eller CNV NORMAL-1001666-1 #NORMAL-1001772-1 #DME-1805818-2
    

#Gammel import af billeder
'''
path = '/Users/tinajensen/Documents/AAU/8 Semester/Projekt/TinaPre/CellData/OCT/train/NORMAL'
#path = '/Users/tinajensen/Documents/AAU/8 Semester/Projekt/TinaPre/CellData/OCT/train/DRUSEN'
#path = '/Users/tinajensen/Documents/AAU/8 Semester/Projekt/TinaPre/CellData/OCT/train/DME'
#path = '/Users/tinajensen/Documents/AAU/8 Semester/Projekt/TinaPre/CellData/OCT/train/CNV'
filename = os.path.join(path, 'NORMAL-1001772-1.JPEG')

test_im = io.imread(filename)
'''

#TINA (NÆSTE 7 LINJER):
img_dir = "/Users/tinajensen/Desktop/TestBilleder/" 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1,0) #0 så det er gray scale
    data.append(img)
    #plt.imshow(img,cmap='gray') #cmap lig gray scale så de printes uden default color map
    #plt.show()


#test_im er float
    
    
#TINA (NÆSTE 3 LINJER):
processedData = []
for img in data:
    test_im = img/(2**(8)-1) # normaliserer
    '''
    test_med = sc.medfilt(test_im, 9)
    
    plt.figure;plt.subplot(1,2,1);plt.imshow(test_im);plt.subplot(1,2,2);plt.imshow(test_med)
    '''
    
    # Anisotropic diffusions filtrering, kappa = 50, iteration n = 200
    filtered_im = ppu.anisodiff(test_im, niter=200)
    #'''
    # DoG 
    gaus1 = gaussian(filtered_im, sigma = 0.4)
    gaus2 = gaussian(filtered_im, sigma = 0.6)
    
    dog = gaus2-gaus1
    
    
    
    dog = dog < -0.00025; #io.imshow(dog)

    
    # Use the value as weights later
    weights = test_im[dog] / float(test_im.max())
    
    # Recasts DoG to int
    dog = dog*1
    
    # Label regioner
    dog_label = ms.label(dog)
    '''
    For hver region, find start raekke og soejle og vurdér om regionen er (1) er stoerre
    end 2000 pixels, (2) om den har pixel vaerdier indenfor de foerste 100 raekker
    og (3) om der er pixel vaerdier indenfor de sidste 100 raekker. 
    Er et af disse opfyldt, så slet regionen i det filtrerede billede
    '''
    
    
    for pred_region in ms.regionprops(dog_label):
        minr, minc, maxr, maxc = pred_region.bbox
        if pred_region.area < 2000 or minr < 100 or maxr > len(dog_label)-1:
            for i in range(minr, maxr):
                for j in range(minc, maxc):
                    dog[i][j] = 0
    
    
    
    '''
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(dog,kernel,iterations = 1)
    plt.figure(4);plt.imshow(dilation);
    '''
    
    
    
    
    
    
    # Get coordinates of pixels corresponding to marked region
    X = np.argwhere(dog)
    
    # Column indices
    x = X[:, 1].reshape(-1, 1)
    # Row indices to predict. Note origin is at top left corner
    y = X[:, 0]
    
    # Ridge regression, i.e., least squares with l2 regularization. 
    # Should probably use a more numerically stable implementation, 
    # e.g., that in Scikit-Learn
    # alpha is regularization parameter. Larger alpha => less flexible curve
    alpha = 0.01
    
    # Construct data matrix, A
    order = 2
    A = ppu.feature(x, order)
    # w = inv (A^T A + alpha * I) A^T y
    w_unweighted = np.linalg.pinv( A.T.dot(A) + alpha * np.eye(A.shape[1])).dot(A.T).dot(y)
    
    
    # Generate test points
    n_samples = test_im.shape[1]
    x_test = np.linspace(0, test_im.shape[1]-1, n_samples, dtype = int)
    X_test = feature(x_test, order)
    # Predict y coordinates at test points
    y_test_unweighted = X_test.dot(w_unweighted)
    
    
    
    ##!!!!!!!!! ALTERNATIV Metode - RANSAC !!!!!!!!!!!!!!##
    dog_zeros = np.nonzero(dog)
    
    
    bestfit = ppu.ransac_polyfit(dog_zeros[1], dog_zeros[0])
    poly = np.poly1d(bestfit)
    
    x = np.linspace(0, test_im.shape[1], n_samples)
    y = poly(x_test)
    
    # Display
    '''fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(test_im)
    ax.plot(x_test, y_test_unweighted, color="green", marker='o', label="Unweighted")
    ax.plot(x, y, color="blue", marker='o', label="RANSAC")
    fig.legend()'''
    
    
    #fig.savefig("curve.png")
    
    #plt.close()
    y_diff = np.diff(np.round(y))
    #y_diffR = np.round(y_diff)
    
    
    
    test_new = np.copy(test_im)
    n_roll = []
    for i in range(test_im.shape[1]-1):
        temp = int(np.round(y[i])-np.round(y[i+1]))
        n_roll.append(temp)
        #Roll statement
        test_new[:,i+1] = np.roll(test_new[:,i+1], temp)
        y[i+1] = y[i+1] + temp
    
    
    
    #'''

    #TINA (NÆSTE 6 LINJER):
    processedData.append(test_new)


d=0
for test_new in processedData:
    filename = "/Users/tinajensen/Desktop/GemtePreproccBilleder/image_%d.jpg"%d #billeder hedder image_0, image_1 osv. Kan bare ændres her
    skimageio.imsave(filename, test_new)
    d+=1


