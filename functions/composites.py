import numpy as np
from skimage import io

def composite_masks(mask_collection, loud=False):
    composite = np.zeros(mask_collection[0].shape[0:2])
    
    for i in range(len(mask_collection)):
        composite += (mask_collection[i] > 0) * (i+1)
    
    if loud:
        io.imshow(composite, cmap='nipy_spectral')
    
    return composite

def decompose_composite(composite):
    mask_coll = []
    
    for color in np.unique(composite):
        if color != 0:
            mask_coll.append(composite == color)
    
    return mask_coll
        
def opt_composite_masks(mask_collection, loud=False):
    composite = np.array(mask_collection)
    depth = len(mask_collection)
    end = depth + 1
    
    color_coef = np.arange(1, end)
    color_coef = color_coef.reshape((depth, 1, 1))
    
    composite = composite * color_coef / 255. 
    composite = np.sum(composite, axis=0)
    
    if loud: 
        io.imshow(composite, cmap='nipy_spectral')
        
    return composite
