from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
from functions.sizes import compute_avg_size

def rescale_composite(image, composite, target_size, loud=False):
    """
    Function which takes image, ground truth, and target size
    and returns a new image with a new composite, each of which
    is rescaled to a new size that approximates the requested 
    target size.
    
    """
    curr_size = compute_avg_size(composite)
    ratio = target_size / curr_size
    ratio = ratio ** (1/2)
    new_shape = (int(composite.shape[0] * ratio),
                 int(composite.shape[1] * ratio))
    
    rescaled_comp = rescale(composite, ratio, anti_aliasing=False,
                            preserve_range=True, order=0)
    resized_img = resize(image, new_shape, anti_aliasing=True,
                          preserve_range=False)
    
    if loud:
        plt.imshow(resized_img); plt.figure()
        plt.imshow(rescaled_comp)
        new_size = compute_avg_size(rescaled_comp)
        print("Original size: %f, Rescaled size: %f" %(curr_size, new_size))
        
    return resized_img, rescaled_comp
        