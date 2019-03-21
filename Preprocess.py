# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:55:04 2019

@author: danie
"""


import os
import PreProcess_Util as ppu
from skimage import io
from skimage.feature import blob_dog
from skimage.filters import gaussian
import skimage.morphology as mp
import skimage.measure as ms
import numpy as np
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
#plt.switch_backend('Agg')
import matplotlib.patches as mpatches

# NORMAL, DRUSEN, DME eller CNV NORMAL-1001666-1 #NORMAL-1001772-1

path = 'C:\\Users\\danie\\Desktop\\Misc\\ST7\\OCT---Transferlearning\\'
filename = os.path.join(path, 'DRUSEN-1001666-1.JPEG')
#filename = 'C:\\Users\\danie\\Desktop\\ST8\\Projekt\\Data\\NORMAL-1001666-1.jpg'
test_im = io.imread(filename)
test_im = test_im/(2**(8)-1) # normaliserer

# Aniso diffusions filtrering, kappa = 50, iteration n = 200
filtered_im = ppu.anisodiff(test_im, niter=200)

# DoG 
gaus1 = gaussian(filtered_im, sigma = 0.4)
gaus2 = gaussian(filtered_im, sigma = 0.6)

dog = gaus2-gaus1

plt.figure;plt.subplot(1,2,1);plt.imshow(dog)

dog = dog < -0.00025; io.imshow(dog)
plt.figure;plt.imshow(dog, 'gray', interpolation = 'none');plt.imshow(mp.skeletonize(dog), 'jet', interpolation = 'none', alpha = 0.5)
#dog_im = blob_dog(filtered_im, threshold=.001, max_sigma=900)

# Use the value as weights later
weights = test_im[dog] / float(test_im.max())

# Recasts DoG to int
dog = dog*1

# Label regioner
dog_label = ms.label(dog)
'''
For hver region, find start raekke og soejle og vurdér om regionen er (1) er større
end 2000 pixels, (2) om den har pixel vaerdier indenfor de foerste 100 raekker
og (3) om der er pixel vaerdier indenfor de sidste 100 raekker. 
Er et af disse opfyldt, så slet regionen i det filtrerede billede

'''
for pred_region in ms.regionprops(dog_label):
    minr, minc, maxr, maxc = pred_region.bbox
    if pred_region.area < 2000 or minr < 100 or maxr > len(dog_label)-100:
        for i in range(minr, maxr):
            for j in range(minc, maxc):
                dog[i][j] = 0

plt.subplot(1,2,2);plt.imshow(dog);


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
x_test = np.linspace(0, test_im.shape[1], n_samples)
X_test = feature(x_test, order)

# Predict y coordinates at test points
y_test_unweighted = X_test.dot(w_unweighted)



##!!!!!!!!! ALTERNATIV Metode - RANSAC !!!!!!!!!!!!!!##
dog_zeros = np.nonzero(dog)


bestfit = ppu.ransac_polyfit(dog_zeros[1], dog_zeros[0])
poly = np.poly1d(bestfit)

x = np.linspace(0, test_im.shape[1], n_samples)
y = poly(x)

# Display
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(test_im)
ax.plot(x_test, y_test_unweighted, color="green", marker='o', label="Unweighted")
ax.plot(x, y, color="blue", marker='o', label="RANSAC")
fig.legend()
#fig.savefig("curve.png")
#'''
plt.close()
y_diff = np.diff(np.round(y))
#y_diffR = np.round(y_diff)

plt.figure;plt.subplot(1,2,1);plt.imshow(test_im);
test_new = test_im
n_roll = []
for i in range(test_im.shape[1]-1):
    temp = int(np.round(y[i])-np.round(y[i+1]))
    n_roll.append(temp)
    test_new[:,i+1] = np.roll(test_new[:,i+1], temp)
    y[i+1] = y[i+1] + temp

plt.subplot(1,2,2);plt.imshow(test_new)
#plt.figure;plt.imshow(test_new)


'''    
Roll with a negative shift:

x2[:, 0] = np.roll(x2[:, 0], -2)
Roll with a positive shift:

x2[:, 4] = np.roll(x2[:, 4], 2)
'''