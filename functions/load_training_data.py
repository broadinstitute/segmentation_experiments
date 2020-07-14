import numpy as np
from skimage import io
from functions.load_all_data import load_data_by_color
from functions.composites import composite_masks
from functions.sizes import compute_avg_size
from functions.crop_image import random_crop

training_imgs_dir = "/raid/data/BBBC038/training/"

# default_imgs, default_mask_colls = load_data_by_color("Default")[::2]
# purple_imgs, purple_mask_colls = load_data_by_color("Purple")[::2]
# tissueBW_imgs, tissueBW_mask_colls = load_data_by_color("TissueBW")[::2]
# pink_imgs, pink_mask_colls = load_data_by_color("Pink-Purple")[::2]

def training_samples(num_samples=1000, colors=[1,1,0,0]):
    """
    Function which loads all training data
    
    User specifies the number of desired samples and which colors to include
    
    colors[0] --> default images
    colors[1] --> purple images
    colors[2] --> black + white images
    colors[3] --> pink-purple images
    """
    
    # only load colors which have been pre-selected
    if colors[0]:
        default_imgs, default_mask_colls = load_data_by_color("Default")[:2]
    if colors[1]:
        purple_imgs, purple_mask_colls = load_data_by_color("Purple")[:2]
    if colors[2]:
        tissueBW_imgs, tissueBW_mask_colls = load_data_by_color("TissueBW")[:2]
    if colors[3]:
        pink_imgs, pink_mask_colls = load_data_by_color("Pink-Purple")[:2]
    
    image_samples = []
    image_labels = []
    
    for i in range(num_samples):
        # load a random image from each selected category and take a random crop
        # add the image and its label to the appropriate lists
        
        if colors[0] == 1:
            def_index = np.random.randint(0, len(default_imgs))
            default_img = default_imgs[def_index]
            default_img = random_crop(default_img)
            
            image_samples.append(default_img)
            image_labels.append(0.)
            
        if colors[1] == 1:
            purp_index = np.random.randint(0, len(purple_imgs))
            purple_img = purple_imgs[purp_index]
            purple_img = random_crop(purple_img)
            
            image_samples.append(purple_img)
            image_labels.append(1.)
            
        if colors[2] == 1:
            bw_index = np.random.randint(0, len(tissueBW_imgs))
            bw_img = tissueBW_imgs[bw_index]
            bw_img = random_crop(bw_img)
            
            image_samples.append(bw_img)
            image_labels.append(2.)
            
        if colors[3] == 1:
            pink_index = np.random.randint(0, len(pink_imgs))
            pink_img = pink_imgs[pink_index]
            pink_img = random_crop(pink_img)
            
            image_samples.append(pink_img)
            image_labels.append(3.)
            
    # convert lists to np arrays
    # normalize samples so each pixel lies between 0 and 1
    image_samples = np.array(image_samples)
    image_samples = image_samples / 255. 
    
    image_labels = np.array(image_labels)

    return image_samples, image_labels
    

    