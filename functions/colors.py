import numpy as np
import skimage
from skimage import io, transform, exposure, data, color
from skimage.color import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def unmix_purple_img(purp_img, loud=False):
    """
    Accepts a purple image object as a parameter 
    and returns the image with the colors unmixed for
    easier segmentation
    """
    
    hematoxylin_matrix = np.ones((3,3)) * (0.644, 0.717, 0.267)       # cell profiler matrix for purple images
    stain_img = purp_img[:, :, [0, 1, 2]]                              # need only first 3 channels to separate stains
    separated_img = separate_stains(stain_img, hematoxylin_matrix)    # apply stain matrix to image
    
    if loud:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,8))
        
        ax[0].set_title("Original")
        ax[0].imshow(purp_img)
        
        ax[1].set_title("Hematoxylin")
        ax[1].imshow(separated_img[:, :, 0])
    
    return separated_img[:, :, 0]

def unmix_pink_imgs(pink_img, loud=False):
    """
    Same as unmix_purple_img but takes a pink image
    as a parameter
    """
    stain_img = pink_img[:, :, [0, 1, 2]]
    separated_img = separate_stains(stain_img, rbd_from_rgb)
    
    if loud:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,8))
        
        ax[0].set_title("Original")
        ax[0].imshow(pink_img)
        
        ax[1].set_title("RBD")
        ax[1].imshow(separated_img[:, :, 1])
        
    return separated_img[:, :, 1]
