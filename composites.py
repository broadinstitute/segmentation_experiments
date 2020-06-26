import numpy as np
from skimage import io

def composite_masks(mask_collection, loud=False):
    composite = 0
    
    for i in range(len(mask_collection)):
        composite += mask_collection[i] * (5 * np.random.sample()) % 255
    
    if loud:
        io.imshow(composite, cmap='viridis')
    
    return composite
