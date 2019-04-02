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
# DRUSEN-1047803-4
# DME-306172-20
# DRUSEN.JPG

path = 'C:\\Users\\danie\\Desktop\\Misc\\ST7\\OCT---Transferlearning\\'
filename = os.path.join(path, 'DRUSEN-1047803-4.JPEG')
#filename = 'C:\\Users\\danie\\Desktop\\ST8\\Projekt\\Data\\NORMAL-1001666-1.jpg'
test_im = io.imread(filename)
test_im = test_im/(2**(8)-1) # normaliserer
'''
test_med = sc.medfilt(test_im, 9)

plt.figure;plt.subplot(1,2,1);plt.imshow(test_im);plt.subplot(1,2,2);plt.imshow(test_med)
'''
'''
# TEST om at rulle billedet inden vi starter for at kunne have et smart bund og top threshold
# Det virker ikke lige nu, da der ikke virker til at vaere system i om det hvide
# er i bunden og at billedet saa skal rykkes ned eller lignende...
# Hvis dette virker, saa husk at aendre threshold for laveste till len -100 igen
Roll_up = False
if test_im[0][int(np.round(test_im.shape[1]/2))] == 1:# or test_im[0][test_im.shape[1]-1] == 1:
    Roll_up = True
    for i in range(test_im.shape[1]-1):
        test_im[:,i+1] = np.roll(test_im[:,i+1], 100)
if Roll_up == False:
    if test_im[test_im.shape[0]-1][int(np.round(test_im.shape[1]/2))] == 1: #or test_im[test_im.shape[0]-1][test_im.shape[1]-1]:
        for i in range(test_im.shape[1]-1):
            test_im[:,i+1] = np.roll(test_im[:,i+1], -100)
'''
'''
Heller ikke helt robust
# Fjerner stoerre nederste regioner, hvis der er nogle
# Ikke alle regioner fjernest saaledes de kan sorteres fra baseret paa areal
# Maaske med et lavere threshhold (Lige nu koerer vi rent 1)
for pred_region in ms.regionprops(ms.label(test_im)):
    minr, minc, maxr, maxc = pred_region.bbox
    if maxr >len(test_im)-1:
        # Finder koordinater for regionen
        Coord = pred_region.coords
        Coord1 = Coord[:, 0]
        Coord2 = Coord[:, 1]
        for i in range(len(Coord1)):
            # Fjerner nu baade regionerne i thresholded billede OG regionsbilleder
            test_im[Coord1[i]][Coord2[i]] = 0
'''

# Anisotropic diffusions filtrering, kappa = 50, iteration n = 200
filtered_im = ppu.anisodiff(test_im, niter=200)
plt.figure;plt.imshow(filtered_im)
#'''
# DoG 
gaus1 = gaussian(filtered_im, sigma = 0.4)
gaus2 = gaussian(filtered_im, sigma = 0.6)

dog = gaus2-gaus1



dog = dog < -0.00018; #io.imshow(dog)
'''
plt.figure(0)
plt.imshow(dog, 'gray', interpolation = 'none')
plt.imshow(mp.skeletonize(dog), 'jet', interpolation = 'none', alpha = 0.5)
'''

# Use the value as weights later
weights = test_im[dog] / float(test_im.max())

# Recasts DoG to int
dog = dog*1

# Fjerner de yderste 5 soejler af pixels 
dog[:,0:5] = 0
dog[:,dog.shape[1]-5:dog.shape[1]] = 0

# Strukturelt element i diskfrom med stoerrelse 3
selem = mp.disk(3)

# Dilation operation
dog_dilate = mp.dilation(dog, selem);

# Label regioner
dog_label = ms.label(dog_dilate)

'''
For hver region, find start raekke og soejle og vurdér om regionen er (1) er stoerre
end 2000 pixels, (2) om den har pixel vaerdier indenfor de foerste 100 raekker
og (3) om der er pixel vaerdier indenfor de sidste 100 raekker. 
Er et af disse opfyldt, sae slet regionen i det filtrerede billede
!!! NYT !!! Nu slettes der ogsae i regions billedet med henblik paa en sekundaer fjerning af billeder senere
'''
# Note! Kan effektiviseres
plt.figure(1);plt.subplot(1,2,1);plt.imshow(dog)
for pred_region in ms.regionprops(dog_label):
    # Finder koordinater for regionen
    Coord = pred_region.coords
    Coord1 = Coord[:, 0]
    Coord2 = Coord[:, 1]
    minr, minc, maxr, maxc = pred_region.bbox
    if pred_region.area < 2000 or minr < 100 or maxr >len(dog_label)-20:
         # Fjerner nu baede regionerne i thresholded billede OG regionsbilleder
         # Nu uden forloekke
         dog[Coord1, Coord2] = 0
         dog_label[Coord1, Coord2] = 0

# Laver en regions variabel NOTE! Kan ogsaa bruges til at goere ovenstaende 
# for loop mere forstaelig
pred_region = ms.regionprops(dog_label)


'''
Lav liste med alle centroider find max af liste samt index og brug det index til at 
slette alle andre regioner undtagen dem som er indenfor 50 pixels vertikalt
'''
# Centroide array, samt max og Index
cent_array = []
for i in range(len(pred_region)):
    cent_array.append(pred_region[i].centroid[0])
Max = np.max(cent_array)    
idx = np.argmax(cent_array)

'''
Hvis Max-index er = iterations variable og laengde af centroide arrayet har samme laengde
Saa slut for loop
Hvis iterations variabel er stoerre eller lig med index og laengden af centroid variablen er 
laengere end iterationsvariablen, da laeg en til iterationsvariablen, saaledes vi
fortsaetter loopet indtil enden af centroide arrayet MEN IKKE SLETTER regionen ved max index
Slutteligt slettes regioner hvis de vertikalt afviger mere end 50 pixels (kan aendres)

'''
for i in range(len(cent_array)-1):
    if i == idx and len(cent_array-1) == idx:
        break
    if i >= idx and len(cent_array-1) > i:
        i = i + 1
    if np.abs(pred_region[i].centroid[0]-Max) >50:
        Coord = pred_region[i].coords
        Coord1 = Coord[:, 0]
        Coord2 = Coord[:, 1] 
        # Nu uden forloekke fjerner vi regionerne i oprindelige billede
        dog[Coord1, Coord2] = 0 

plt.subplot(1,2,2);plt.imshow(dog);


# Get coordinates of pixels corresponding to marked region
X = np.argwhere(dog)

''' Column indices     !!!! NOTE  Denne laver stadig en x-vektor, som ikke er integers!!!!
Dette mener jeg blev fikset i jeres version, men det volder ikke problemer pt. ellers tjek
Preprocess_Centroid
'''
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

'''
Saenker iterativt noedvendige antal punkter i parabelfittet indtil vi har et bestfit
'''
for i in reversed(range(50)):
    bestfit = ppu.ransac_polyfit(dog_zeros[1], dog_zeros[0], n = i)
    if bestfit is not None:
        check = i
        break
        

poly = np.poly1d(bestfit)

x = np.linspace(0, test_im.shape[1], n_samples)
y = poly(x)

# Display
'''fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(test_im)
ax.plot(x_test, y_test_unweighted, color="green", marker='o', label="Unweighted")
ax.plot(x, y, color="blue", marker='o', label="RANSAC")
fig.legend()'''
# Specificerer stoerrelse til print, kan undvaeres
fig1 = plt.figure(2, figsize = (496*1.3/139, 512*1.3/139), dpi = 139);
plt.axis('off');
plt.imshow(test_im);
plt.plot(x_test, y_test_unweighted, color="green", marker='o', markersize =2, label="Unweighted");
plt.plot(x, y, color="blue", marker='o', markersize =2, label="RANSAC")

#fig1.savefig("RANSAC2.png")

#plt.close()
y_diff = np.diff(np.round(y))
#y_diffR = np.round(y_diff)

'''
Det kan ogsaa optimeres ved at tage forskellen paa foerste reference pixel og de resterende pixels
Herefter kan man give roll en tuple istedet for en for loekke, maaske
testtest = np.copy(test_im)
n_roll_test = np.round(Y_test[0]) - np.round(Y_test[1:len(y)])
n_roll_test = n_roll_test.astype(int)
testtest = np.roll(testtest[:,1:len(testtest)], n_roll_test, axis = 1)

# Der er stadig et problem med at kun halvdelen af billedet flyttes.
'''
test_new = np.copy(test_im)
n_roll = []
Y_test = np.copy(y)

'''
Ny forbedret rulle metode, resultat er det samme
sammenlign test_new med testtest, hvis i vil vaere sikre
'''
testtest = np.copy(test_im)
n_roll_test = np.round(Y_test[0]) - np.round(Y_test[1:len(y)])
n_roll_test = n_roll_test.astype(int)
x_col = np.linspace(0, test_im.shape[1]-1, n_samples); x_col = x_col.astype(int)
cols = x_col[1:len(x_col)]  # Columns to be rolled
dirn = -n_roll_test # Offset with direction as sign
n = testtest.shape[0]
testtest[:,cols] = testtest[np.mod(np.arange(n)[:,None] + dirn,n),cols]

'''
for i in range(test_im.shape[1]-1):
    temp = int(np.round(y[i])-np.round(y[i+1]))
    n_roll.append(temp)
    #Roll statement
    test_new[:,i+1] = np.roll(test_new[:,i+1], temp)
    y[i+1] = y[i+1] + temp
'''
plt.figure(3);plt.subplot(1,2,1);plt.imshow(test_im);
plt.subplot(1,2,2);plt.imshow(testtest)
#plt.figure;plt.imshow(test_new)

#'''