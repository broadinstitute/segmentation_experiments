import numpy as np
from skimage import io

training_imgs_dir = "/raid/data/BBBC038/training/"

def compute_avg_size(img_name, loud=False):
    data_dir = f"{training_imgs_dir}{img_name}/masks/*.png"
    mask_coll = io.collection.ImageCollection(data_dir)
    size = 0
    i = 0
    
    for mask in mask_coll:
        size += np.sum(mask != 0)
        if loud: 
            print("Nucleus #%d has size %d" %(i + 1, np.sum(mask!=0)))
        i += 1
        
    avg_size = size / len(mask_coll) 
    if loud: 
        print("Average nucleus size is %f" %(avg_size))
    
    return avg_size
