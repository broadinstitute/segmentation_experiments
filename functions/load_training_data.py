import numpy as np
from skimage import io
from functions.load_all_data import load_data_by_color
from functions.composites import composite_masks
from functions.sizes import compute_avg_size
from functions.crop_image import random_crop

# Create load function which returns full images, mask composites, and image labels.
# Do not need to calculate average size in load function
def training_samples(num_samples=1000, colors=[1,1,0,0], load_masks=False):
    """
    Function which loads all training data
    
    User specifies the number of desired samples and which colors to include
    
    Includes optional paramter to load mask composites as well
    
    colors[0] --> default images
    colors[1] --> purple images
    colors[2] --> black + white images
    colors[3] --> pink-purple images
    """
    image_samples = []
    image_labels = []
    mask_composites = []
    full_images = []
    
    # only load colors which have been pre-selected
    # load masks regardless of load_mask parameter to avoid calling load twice
    if colors[0]:
        default_imgs, default_mask_colls = load_data_by_color("Default")[:2]
    if colors[1]:
        purple_imgs, purple_mask_colls = load_data_by_color("Purple")[:2]
    if colors[2]:
        tissueBW_imgs, tissueBW_mask_colls = load_data_by_color("TissueBW")[:2]
    if colors[3]:
        pink_imgs, pink_mask_colls = load_data_by_color("Pink-Purple")[:2]
        
    for i in range(num_samples):
        # load a random image from each selected category and take a random crop
        # add the image and its label to the appropriate lists
        # if user opts to load masks, load the corresponding mask from collection
        # keep all of the if statements inside the for loop to aid readablility
        
        if colors[0]:
            def_index = np.random.randint(0, len(default_imgs))
            default_img = default_imgs[def_index]
            cropped_default_img = random_crop(default_img)[0]

            image_samples.append(cropped_default_img)
            image_labels.append(0.)
            
            if load_masks:
                mask_comp = composite_masks(default_mask_colls[def_index])
                mask_composites.append(mask_comp)
                full_images.append(default_img)

        if colors[1]:
            purp_index = np.random.randint(0, len(purple_imgs))
            purple_img = purple_imgs[purp_index]
            cropped_purple_img = random_crop(purple_img)[0]

            image_samples.append(cropped_purple_img)
            image_labels.append(1.)
            
            if load_masks:
                mask_comp = composite_masks(purple_mask_colls[purp_index])
                mask_composites.append(mask_comp)
                full_images.append(purple_img)

        if colors[2]:
            bw_index = np.random.randint(0, len(tissueBW_imgs))
            bw_img = tissueBW_imgs[bw_index]
            cropped_bw_img = random_crop(bw_img)[0]

            image_samples.append(cropped_bw_img)
            image_labels.append(2.)
            
            if load_masks:
                mask_comp = composite_masks(tissueBW_mask_colls[bw_index])
                mask_composites.append(mask_comp)
                full_images.append(bw_img)

        if colors[3]:
            pink_index = np.random.randint(0, len(pink_imgs))
            pink_img = pink_imgs[pink_index]
            cropped_pink_img = random_crop(pink_img)[0]

            image_samples.append(cropped_pink_img)
            image_labels.append(3.)
            
            if load_masks:
                mask_comp = composite_masks(pink_mask_colls[pink_index])
                mask_composites.append(mask_comp)
                full_images.append(pink_img)
        
    # convert lists to np arrays
    # normalize samples so each pixel lies between 0 and 1
    image_samples = np.array(image_samples)
    image_samples = image_samples / 255. 
    image_labels = np.array(image_labels)

    if load_masks:
        full_images = np.array(full_images)
        return image_samples, image_labels, mask_composites, full_images
    else:
        return image_samples, image_labels

# Another function is for cropping
# receives loaded data (images, composites, labels)
# returns data set with the same number of examples for each class label
# each entry will have the cropped image, cropped mask, class label, and size

def load_regression_samples(images, labels, composites):
    """
    Function which returns cropped images, masks, nuclei sizes, and class labels
    
    For each image, composite, label: 
        takes a random crop of the image and mask
        computes the average nuclei size in the cropped mask
        appends results to their respective arrays
    
    """
    cropped_imgs = []
    cropped_masks = []
    nuclei_sizes = []
    num_samples = len(labels)
    
    for i in range(num_samples):
        cropped_image, cropped_mask = random_crop(img=images[i], 
                                                 mask_composite=composites[i],
                                                 crop_size=64, loud=False)
        size = compute_avg_size(cropped_mask)
        
        cropped_imgs.append(cropped_image)
        cropped_masks.append(cropped_mask)
        nuclei_sizes.append(size)
        
    cropped_imgs = np.array(cropped_imgs) / 255.
    cropped_masks = np.array(cropped_masks)
    nuclei_sizes = np.array(nuclei_sizes)
        
    return cropped_imgs, cropped_masks, nuclei_sizes, labels