#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:54:21 2019

@author: tinajensen
"""

import cv2
import os, os.path
import glob
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from PIL import Image as PImage

from datetime import datetime
import uuid

from skimage import io as skimageio


#LOAD DATA

''' 
MULIGHED 1 - liste
Nedenstående kode virker som udgangspunkt. 
imgs er billederne, en liste, type er image
Man skal dog lave en path for hver af de fire mapper (DME, CNV, DRUSEN,NORMAL)
Og så er der lige det faktum at min computer i hvert fald ikke kunne åbne
train mapperne, fordi der var for mange billeder. Den stoppede efter 4 min 
Virker dog fint med test
'''

'''
def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = PImage.open(path + image)
        loadedImages.append(img)

    return loadedImages

path = "/Users/tinajensen/Documents/AAU/8 Semester/Projekt/TinaPre/CellData/OCT/test/NORMAL/"

# your images in an array
imgs = loadImages(path)


#for img in imgs:
    #img.show()
'''




'''
MULIGHED 2 - openCV, den vi bruger
Nedenstående virker også.
data er en liste med billederne
img er billederne enkeltvis, type uint8
Virker på test billeder, men den får også cancer når jeg gør det ved træning
Kørte en halv time og så var der ikke mere plads på min computer (40GB)
På 4 min kan den loade 10.000 billeder (der er 50.000 i træning normal), men bliver lidt langsom

OBS: Billedepakken er cv og andre steder bruger vi skimage
'''

#Testbilleder er en mappe med tre billeder: DME-1001666-1.jpeg DRUSEN-1001666-1.jpeg NORMAL-1001666-1.jpeg
img_dir = "/Users/tinajensen/Desktop/TestBilleder" 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1,0) #0 så det er gray scale
    data.append(img)


    #plt.imshow(img,cmap='gray') #cmap lig gray scale så de printes uden default color map
    #plt.show()


#GEM DATA (IKKE LABEL NAVN) - IKKE DET VI SKAL
'''
d = 0
for img in data:
    filename = "/Users/tinajensen/Desktop/GemteBilleder/image_%d.jpg"%d #billeder hedder image_0, image_1 osv. Kan bare ændres her
    cv2.imwrite(filename, img)
    d+=1
'''


'''
d=0
for img in data:
    filename = "/Users/tinajensen/Desktop/GemteBilleder/image_%d.jpg"%d #billeder hedder image_0, image_1 osv. Kan bare ændres her
    skimageio.imsave(filename, img)
    d+=1
'''    


#GEM DATA (LABEL NAVN) - DET HER VI SKAL

#MANGLER:
#fix problem med rigtig billede/navn med indeksering 
#gemme i new folder - vælg mappe hvor den skal gemme, ikke fokus filnavn
    

'''
#alle billeder med alle navne
d=0
for img in data:
    for f1 in files:
        filenameSave = f1.replace('.jpeg', 'preprocessed_%d.jpeg') %d
        skimageio.imsave(filenameSave,img)
        d+=1
'''


'''
#Rigtig billeder, samme navn
d=0
for img in data:
    filenameSave = f1.replace('.jpeg', 'preprocessed_%d.jpeg') %d
    skimageio.imsave(filenameSave,img)
    d+=1
'''

'''
#Rigtig navn, samme billede for alle
for f1 in files:
    filenameSave = f1.replace('.jpeg', 'preprocessed.jpeg')
    skimageio.imsave(filenameSave,img)
'''



#Udgangspunkt i rigtig navn, samme billede for alle
for f1 in files:
    filenameSave = f1.replace('.jpeg', 'preprocessed.jpeg')
    skimageio.imsave(filenameSave,data[:])

    #så skriver den problem med dtype, kan fixes ved:
    #arrData = np.array(data)
    #Og så skriv aeeData hvor der nu står data i imsave
    #Men så siger den for mange input channels, fordi der er 3 billeder...
    #ooog så er jeg tilbage ved brug for to for løkker og det virker ikke...







