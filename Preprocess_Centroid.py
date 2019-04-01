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
import matplotlib.pyplot as plt


def feature(x, order=2):
    """Generate polynomial feature of the form
    [1, x, x^2, ..., x^order] where x is the column of x-coordinates
    and 1 is the column of ones for the intercept.
    """
    x = x.reshape(-1, 1)
    return np.power(x, np.arange(order+1).reshape(1, -1)) 

# NORMAL, DRUSEN, DME eller CNV NORMAL-1001666-1 #NORMAL-1001772-1 #DME-1805818-2

#path = 'C:\\Users\\MetteToettrupGade\\Desktop\\CellData\\OCT\\train\\DME\\'
path = 'C:\\Users\\MetteToettrupGade\\Desktop\\CellData\\OCT\\train\\DRUSEN\\'
filename = os.path.join(path, 'DRUSEN-538779-13.JPEG')
train_im = io.imread(filename)
plt.figure(0);plt.subplot(1,2,1);plt.imshow(train_im)
train_im = train_im/(2**(8)-1) # normaliserer
'''
test_med = sc.medfilt(test_im, 9)

plt.figure;plt.subplot(1,2,1);plt.imshow(test_im);plt.subplot(1,2,2);plt.imshow(test_med)
'''

# Anisotropic diffusions filtrering, kappa = 50, iteration n = 200
filtered_im = ppu.anisodiff(train_im, niter=200)
plt.subplot(1,2,2);plt.imshow(filtered_im)
#'''
# DoG 
gaus1 = gaussian(filtered_im, sigma = 0.4)
gaus2 = gaussian(filtered_im, sigma = 0.6)

dog = gaus2-gaus1

dog = dog < -0.00025; #io.imshow(dog)
'''
plt.figure(0)
plt.imshow(dog, 'gray', interpolation = 'none')
plt.imshow(mp.skeletonize(dog), 'jet', interpolation = 'none', alpha = 0.5)
'''

# Use the value as weights later
weights = train_im[dog] / float(train_im.max())


# Recasts DoG to int
dog = dog*1

dog_dilation = np.copy(dog)

'''
For hver region, find start raekke og soejle og vurdér om regionen er (1) er stoerre
end 2000 pixels, (2) om den har pixel vaerdier indenfor de foerste 100 raekker
og (3) om der er pixel vaerdier indenfor de sidste 100 raekker. 
Er et af disse opfyldt, så slet regionen i det filtrerede billede
'''

plt.figure(1);plt.subplot(1,3,1);plt.imshow(dog)


# Label regioner
dog_label = ms.label(dog) #ms.label(img_dilation)

dog_remove = np.copy(dog)

# Distancen mellem regioners centroider
region_prop = ms.regionprops(dog_label)
for pred_region in range(len(region_prop)):
    if pred_region < len(region_prop)-1:
        dist_region = region_prop[pred_region+1].centroid[1] - region_prop[pred_region].centroid[1]
    elif pred_region == len(region_prop)-1:
        dist_region = region_prop[pred_region].centroid[1] - region_prop[0].centroid[1]
    
    # Remove regions med distance     
    Coord = region_prop[pred_region].coords
    Coord1 = Coord[:,0]
    Coord2 = Coord[:,1]
    if abs(dist_region) > 500: 
        for i in range(len(Coord1)):
           dog_remove[Coord1[i]][Coord2[i]] = 0
           
    if abs(dist_region) <= 500:
        selem = mp.disk(4.5)
        img_dilation = mp.dilation(dog_dilation, selem);
        plt.subplot(1,3,2);plt.imshow(img_dilation);

plt.subplot(1,3,3);plt.imshow(dog_remove);

# Flattning starts here (Mette's Method)
# Get coordinates of pixels corresponding to marked region
X = np.argwhere(dog_remove)  #(img_dialtion)

# Column indices
x = X[:, 1].reshape(-1, 1)
# Row indices to predict. Note origin is at top left corner
y = X[:, 0]

''' 
# Mettes metode forsøg med flattening baseret på intensiteter, ikke færdig
i=1
for x in range(i, len(np.unique(x))):
    columDiff = y[0] - y[i]
    flat_arr = np.array(img_dilation)
    #flat_arr[flat_arr < 2] = 0
    flat_arr[:,i] = np.roll(img_dilation[:,i], columDiff)    
    i += 1

plt.figure(2);plt.title('Flattening');plt.imshow(img_dilation);
'''


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
n_samples = train_im.shape[1]
x_train = np.linspace(0, train_im.shape[1], n_samples)
X_train = feature(x_train, order)

# Predict y coordinates at test points
y_train_unweighted = X_train.dot(w_unweighted)



##!!!!!!!!! ALTERNATIV Metode - RANSAC !!!!!!!!!!!!!!##
dog_zeros = np.nonzero(dog) #(img_dialtion)


bestfit = ppu.ransac_polyfit(dog_zeros[1], dog_zeros[0])
poly = np.poly1d(bestfit)

x = np.linspace(0, train_im.shape[1], n_samples)
y = poly(x)


# Display
''' fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(test_im)
ax.plot(x_test, y_test_unweighted, color="green", marker='o', label="Unweighted")
ax.plot(x, y, color="blue", marker='o', label="RANSAC")
fig.legend()'''

plt.figure(2);
plt.imshow(train_im);
plt.plot(x_train, y_train_unweighted, color="green", marker='o', label="Unweighted");
plt.plot(x, y, color="blue", marker='o', label="RANSAC")

#fig.savefig("curve.png")

#plt.close()
y_diff = np.diff(np.round(y))
#y_diffR = np.round(y_diff)



train_new = np.copy(train_im)
n_roll = []
for i in range(train_im.shape[1]-1):
    temp = int(np.round(y[i])-np.round(y[i+1]))
    n_roll.append(temp)
    #Roll statement
    train_new[:,i+1] = np.roll(train_new[:,i+1], temp)
    y[i+1] = y[i+1] + temp

plt.figure(3);plt.subplot(1,2,1);plt.imshow(train_im);
plt.subplot(1,2,2);plt.imshow(train_new)
#plt.figure;plt.imshow(test_new)

#'''