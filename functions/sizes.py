import numpy as np
from skimage import io

training_imgs_dir = "/raid/data/BBBC038/training/"

def compute_avg_size(mask_collection, loud=0):
    size = 0
    i = 0
    
    for mask in mask_collection:
        size += np.sum(mask != 0)
        if loud == 2: 
            print("Nucleus #%d has size %d" %(i + 1, np.sum(mask!=0)))
        i += 1
        
    avg_size = size / len(mask_collection) 
    if loud: 
        print("Average nucleus size is %f" %(avg_size))
    
    return avg_size
