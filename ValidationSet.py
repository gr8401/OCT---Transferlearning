# Creating a test and validation set (80/20) from the original test set

import os
import shutil
import random
### CNV ###
# Define the source folder from where the validation images should be found and removed to another folder
CNV_source = os.listdir('C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\CNV\\')

# Create the folder for the validation images
try:
   os.makedirs('C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\CNV_val\\')
except OSError:
    # The directory already existed, nothing to do
    pass

CNV_val = 'C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\CNV_val\\'

# Randomly, choosing 20 % of the test set and saving is as valdiation set
CNV_number_files = int(len(CNV_source))  # Number of files in folder
CNV_val20 = round(CNV_number_files/100*20)   # Randomly, choosing 20 % 
CNV_sampling = random.sample(CNV_source, k=CNV_val20)

# Moving the random choosen images to the validation folder
for f in CNV_source:
    if f in CNV_sampling:
        shutil.move('C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\CNV\\'+f, CNV_val)


### DME ###
DME_source = os.listdir('C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\DME\\')
try:
   os.makedirs('C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\DME_val\\')
except OSError:
    pass

DME_val = 'C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\DME_val\\'

DME_number_files = int(len(DME_source))  # Number of files in folder
DME_val20 = round(DME_number_files/100*20)   # Randomly, choosing 20 % 
DME_sampling = random.sample(DME_source, k=DME_val20)

for f in DME_source:
    if f in DME_sampling:
        shutil.move('C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\DME\\'+f, DME_val)


### DRUSEN ###
DRUSEN_source = os.listdir('C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\DRUSEN\\')
try:
    os.makedirs('C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\DRUSEN_val\\')
except OSError:
    pass

DRUSEN_val = 'C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\DRUSEN_val\\'

DRUSEN_number_files = int(len(DRUSEN_source)) #Number of files in folder
DRUSEN_val20 = round(DRUSEN_number_files/100*20) #Randomly, choosing 20 %
DRUSEN_sampling = random.sample(DRUSEN_source, k=DRUSEN_val20)

for f in DRUSEN_source:
    if f in DRUSEN_sampling:
        shutil.move('C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\DRUSEN\\'+f, DRUSEN_val)


### NORMAL ###
NORMAL_source = os.listdir('C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\NORMAL') # NORMAL directory path
try:
    os.makedirs('C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\NORMAL_val\\')
except OSError:
    pass

NORMAL_val = 'C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\NORMAL_val\\'

NORMAL_number_files = int(len(NORMAL_source)) #Number of files in folder
NORMAL_val20 = round(NORMAL_number_files/100*20) #Randomly, choosing 20 %
NORMAL_sampling = random.sample(NORMAL_source, k=NORMAL_val20)

for f in NORMAL_source:
    if f in NORMAL_sampling:
        shutil.move('C:\\Users\\MetteToettrupGade\\Desktop\\BillederKodetest\\NORMAL\\'+f, NORMAL_val)