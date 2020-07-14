import numpy as np
from skimage import io

def compute_avg_size(mask_composite, loud=0):
    num_nuclei = len(np.unique(mask_composite)) - 1 
    
    total_size = np.sum(mask_composite != 0)
    avg_size = total_size / num_nuclei
    
    if loud > 0:
        print("Average nuclei size is %f" %(avg_size))
        
        if loud == 2:
            io.imshow(mask_composite, cmap='nipy_spectral')
        
        
    return avg_size