# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:23:14 2019

@author: danie
"""

import numpy as np
import os
import glob
import cv2
import skimage.measure as ms
from skimage.filters import gaussian
from skimage import io
import warnings

def feature(x, order=2):
    """Generate polynomial feature of the form
    [1, x, x^2, ..., x^order] where x is the column of x-coordinates
    and 1 is the column of ones for the intercept.
    """
    x = x.reshape(-1, 1)
    return np.power(x, np.arange(order+1).reshape(1, -1)) 

def prep_im(im, thresh, col_to_remove):   
    # Remove low or top artifacts
    test_im2 = im >0.9

    for pred_region in ms.regionprops(ms.label(test_im2)):
        minr, minc, maxr, maxc = pred_region.bbox
        if maxr >len(im)-1:# or minr == 0:
            # Finder koordinater for regionen
            Coord = pred_region.coords
            '''
            insertvector = np.ones([test_im2.shape[1],2], dtype = int)
            insertvector[:,0] = insertvector[:,0]*(Coord[0,0]-1)
            insertvector[:,1] = np.linspace(0, test_im2.shape[1]-1, test_im2.shape[1])
            Coord = np.append(insertvector, Coord, axis = 0)
            '''
            Coord_1 = Coord.copy()
            Coord_1[:,0] = Coord_1[:,0]-1
            Coord = np.append(Coord_1, Coord, axis = 0)
            
            Coord1 = Coord[:, 0]
            Coord2 = Coord[:, 1]
            im[Coord1, Coord2] = 0
            '''
            test_im3 = im > 0.6
            for pred_region in ms.regionprops(ms.label(test_im3[Coord[0,0]:test_im3.shape[1],:])):
                Coord_rand = pred_region.coords
                im[Coord_rand[:,0], Coord_rand[:,1]] = 0
            '''
    
    # Anisotropic diffusions filtrering, kappa = 50, iteration n = 200
    filtered_im = anisodiff(im, niter=100)
    # DoG 
    gaus1 = gaussian(filtered_im, sigma = 0.4)
    gaus2 = gaussian(filtered_im, sigma = 0.6)
    dog = gaus2-gaus1
    # Thresholding
    dog = dog < thresh
    # Fjerner de yderste 5 soejler af pixels 
    dog[:,0:col_to_remove] = 0
    dog[:,im.shape[1]-col_to_remove:im.shape[1]] = 0
    # Recasts DoG to int
    dog = dog*1
    return dog


def ransac_polyfit(x, y, order=2, n=50, k=20, t=50, d=100, f=0.8):
  # Thanks https://en.wikipedia.org/wiki/Random_sample_consensus
  
  # n – minimum number of data points required to fit the model
  # k – maximum number of iterations allowed in the algorithm
  # t – threshold value to determine when a data point fits a model
  # d – number of close data points required to assert that a model fits well to data
  # f – fraction of close data points required
  
  besterr = np.inf
  bestfit = None
  for kk in range(k):
    maybeinliers = np.random.randint(len(x), size=n)
    maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
    alsoinliers = np.abs(np.polyval(maybemodel, x)-y) < t
    if sum(alsoinliers) > d and sum(alsoinliers) > len(x)*f:
      bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
      thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers])-y[alsoinliers]))
      if thiserr < besterr:
        bestfit = bettermodel
        besterr = thiserr
  return bestfit

def norm_poly_app(input_image, alpha):
    # Get coordinates of pixels corresponding to marked region
    X = np.argwhere(input_image)
    
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
    
    
    # Construct data matrix, A
    order = 2
    A = feature(x, order)
    # w = inv (A^T A + alpha * I) A^T y
    w_unweighted = np.linalg.pinv( A.T.dot(A) + alpha * np.eye(A.shape[1])).dot(A.T).dot(y)
    
    
    # Generate test points
    n_samples = input_image.shape[1]
    x_test = np.linspace(0, input_image.shape[1]-1, n_samples)
    X_test = feature(x_test, order)
    
    # Predict y coordinates at test points
    y_test_unweighted = X_test.dot(w_unweighted)
    return x_test, y_test_unweighted




def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1,ploton=False):
        """
        Anisotropic diffusion.
 
        Usage:
        imgout = anisodiff(im, niter, kappa, gamma, option)
 
        Arguments:
                img    - input image
                niter  - number of iterations
                kappa  - conduction coefficient 20-100 ?
                gamma  - max value of .25 for stability
                step   - tuple, the distance between adjacent pixels in (y,x)
                option - 1 Perona Malik diffusion equation No 1
                         2 Perona Malik diffusion equation No 2
                ploton - if True, the image will be plotted on every iteration
 
        Returns:
                imgout   - diffused image.
 
        kappa controls conduction as a function of gradient.  If kappa is low
        small intensity gradients are able to block conduction and hence diffusion
        across step edges.  A large value reduces the influence of intensity
        gradients on conduction.
 
        gamma controls speed of diffusion (you usually want it at a maximum of
        0.25)
 
        step is used to scale the gradients in case the spacing between adjacent
        pixels differs in the x and y axes
 
        Diffusion equation 1 favours high contrast edges over low contrast ones.
        Diffusion equation 2 favours wide regions over smaller ones.
 
        Reference:
        P. Perona and J. Malik.
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        12(7):629-639, July 1990.
 
        Original MATLAB code by Peter Kovesi  
        School of Computer Science & Software Engineering
        The University of Western Australia
        pk @ csse uwa edu au
        <http://www.csse.uwa.edu.au>
 
        Translated to Python and optimised by Alistair Muldal
        Department of Pharmacology
        University of Oxford
        <alistair.muldal@pharm.ox.ac.uk>
 
        June 2000  original version.      
        March 2002 corrected diffusion eqn No 2.
        July 2012 translated to Python
        """
 
        # ...you could always diffuse each color channel independently if you
        # really want
        if img.ndim == 3:
                warnings.warn("Only grayscale images allowed, converting to 2D matrix")
                img = img.mean(2)
 
        # initialize output array
        img = img.astype('float32')
        imgout = img.copy()
 
        # initialize some internal variables
        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(imgout)
        gE = gS.copy()
 
        # create the plot figure, if requested
        if ploton:
                import pylab as pl
                from time import sleep
 
                fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
                ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
 
                ax1.imshow(img,interpolation='nearest')
                ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
                ax1.set_title("Original image")
                ax2.set_title("Iteration 0")
 
                fig.canvas.draw()
 
        for ii in range(niter):
 
                # calculate the diffs
                deltaS[:-1,: ] = np.diff(imgout,axis=0)
                deltaE[: ,:-1] = np.diff(imgout,axis=1)
 
                # conduction gradients (only need to compute one per dim!)
                if option == 1:
                        gS = np.exp(-(deltaS/kappa)**2.)/step[0]
                        gE = np.exp(-(deltaE/kappa)**2.)/step[1]
                elif option == 2:
                        gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
                        gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
 
                # update matrices
                E = gE*deltaE
                S = gS*deltaS
 
                # subtract a copy that has been shifted 'North/West' by one
                # pixel. don't as questions. just do it. trust me.
                NS[:] = S
                EW[:] = E
                NS[1:,:] -= S[:-1,:]
                EW[:,1:] -= E[:,:-1]
 
                # update the image
                imgout += gamma*(NS+EW)
 
                if ploton:
                        iterstring = "Iteration %i" %(ii+1)
                        ih.set_data(imgout)
                        ax2.set_title(iterstring)
                        fig.canvas.draw()
                        # sleep(0.01)
 
        return imgout
    
def remove_based_centroid(pred_region, image):
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
    minr = np.min(pred_region[idx].coords[:,1])
    maxr = np.max(pred_region[idx].coords[:,1])
    
    '''
    Hvis Max-index er = iterations variable og laengde af centroide arrayet har samme laengde
    Saa slut for loop
    Hvis iterations variabel er stoerre eller lig med index og laengden af centroid variablen er 
    laengere end iterationsvariablen, da laeg en til iterationsvariablen, saaledes vi
    fortsaetter loopet indtil enden af centroide arrayet MEN IKKE SLETTER regionen ved max index
    Slutteligt slettes regioner hvis de vertikalt afviger mere end 50 pixels (kan aendres)
    '''
    for i in range(len(cent_array)-1):
        if i == idx and len(cent_array)-1 == idx:
            break
        if i >= idx and len(cent_array)-1 > i:
            i = i + 1
        if np.abs(pred_region[i].centroid[0]-Max) >50:# or minr < pred_region[i].centroid[1] < maxr:
            Coord = pred_region[i].coords
            Coord1 = Coord[:, 0]
            Coord2 = Coord[:, 1] 
            # Nu uden forloekke fjerner vi regionerne i oprindelige billede
            image[Coord1, Coord2] = 0 
    return image

def roll_place_im(y, im_to_roll):
    '''
    Ny forbedret rulle metode, goer det samme, men med anden metodik
    '''
    testtest = np.copy(im_to_roll)
    n_roll_test = np.round(y[0]) - np.round(y[1:len(y)])
    y_70 = np.round(im_to_roll.shape[0]*0.7) #rækken i billedet der svarer til 70%
    move_bm = np.round(y_70 - y[0]) #forskellen fra poly placering og rækken svarende til 70%
    n_roll_test = n_roll_test + move_bm #samlet flytning af membranen til rækken svarende til 70%
    n_roll_test = n_roll_test.astype(int)
    # Lav vektor mellem 0-laengde a billedet minus 1, som har antal punkter lig med laengden
    x_col = np.linspace(0, im_to_roll.shape[1]-1, im_to_roll.shape[1]); x_col = x_col.astype(int)
    cols = x_col[1:len(x_col)]  # Columns to be rolled
    dirn = -n_roll_test # Offset with direction as sign
    n = testtest.shape[0]
    testtest[:,cols] = testtest[np.mod(np.arange(n)[:,None] + dirn,n),cols]
    return testtest, n_roll_test

def hitpixels(im_regions, x_to_hit, y_to_hit):
    y_to_hit = np.round(y_to_hit)
    y_to_hit = y_to_hit.astype(int)
    x_to_hit = x_to_hit.astype(int)
    if np.abs(np.max(x_to_hit)) > im_regions.shape[1]:
        x_to_hit[np.argwhere(x_to_hit >= im_regions.shape[1])] = im_regions.shape[1]-1
    if np.abs(np.max(y_to_hit)) > im_regions.shape[0]:
        y_to_hit[np.argwhere(y_to_hit >= im_regions.shape[0])] = im_regions.shape[0]-1
    Overlap = im_regions[y_to_hit, x_to_hit]
    hitpixels = np.size(np.where(Overlap ==1))
    return hitpixels

def load(img_dir):
    #img_dir = r'C:\\Users\\danie\\Desktop\\ST8\\Projekt\\Data\\OCT2017\\SaveTest\\' 
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    data = []
    
    for f1 in files:
        img = cv2.imread(f1,0) #0 så det er gray scale
        data.append(img)
    return data, files

def save(filepaths, data, sav_dir):
    #for f1 in range(len(filepaths)):
    filepaths = filepaths.replace('SaveTest', sav_dir)
    #if '-preprocessed' not in files[f1]:
    try:
        os.remove(filepaths.replace('.jpeg', '-preprocessed.jpeg'))
    except FileNotFoundError:
        pass
    io.imsave(filepaths.replace('.jpeg', '-preprocessed.jpeg'),data)