import numpy as np
from skimage import io

def composite_masks(mask_collection, loud=False):
    composite = np.zeros(mask_collection.shape[0:2])
    
    for i in range(len(mask_collection)):
        composite += (mask_collection[i] > 0) * (i+1)
    
    if loud:
        io.imshow(composite, cmap='nipy_spectral')
    
    return composite
