# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:26:19 2019

@author: unnit
"""
import cv2
import glob
import os
import PreProcess_Util as ppu

img_dir = 'C:\\Users\\unnit\\OneDrive\\Desktop\\compare\\'
all_dir = 'C:\\Users\\unnit\\OneDrive\\Desktop\\compare\\CNV_all\\'
pp_dir = 'C:\\Users\\unnit\\OneDrive\\Desktop\\compare\\CNV_pp\\'
destination_dir = 'C:\\Users\\unnit\\OneDrive\\Desktop\\compare\\CNV_destination\\'


# =============================================================================
# # define original text
# CNV_pp = ['image1-preprocessed','image2-preprocessed','image3-preprocessed']
# # define modified text
# CNV_all = ['image1','image2','image3','image4','image5','image6']
# 
# CNV_destination = []
# =============================================================================
CNV_all = os.listdir(all_dir)
CNV_pp = os.listdir(pp_dir)
CNV_destination = os.listdir(destination_dir)



for i in range(len(CNV_all)):
    image = CNV_all[i]
    image = image[:-5]
    for j in range(len(CNV_pp)):
        compare = CNV_pp[j].find(image)
        if compare == 0:
            os.replace(all_dir+image+'.jpeg',destination_dir+image+'.jpeg')
            CNV_destination.append(image)
            break
        else:
            pass
        
