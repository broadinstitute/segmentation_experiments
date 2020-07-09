from skimage import io
import numpy as np
import matplotlib.pyplot as plt


def random_crop(img, crop_size=64, loud=False):
    img = img[:, :, :3]    # neglect alpha value
    row_start = np.random.randint(0, img.shape[0] - crop_size)  # select a random row and column
    col_start = np.random.randint(0, img.shape[1] - crop_size)  # which still allows for a full crop

    row_end = row_start + crop_size
    col_end = col_start + crop_size                            
    
    cropped_img = img[row_start:row_end, col_start:col_end, :]    # slice the image along our desired border
    
    if loud:
        print("Original image size is %d by %d. Crop starts at (%d,%d). Cropped image is %dx%d pixels" 
              %(img.shape[0], img.shape[1], row_start, col_start, crop_size, crop_size))
        
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,8))
        
        ax[0].set_title("Original")
        ax[0].imshow(img)
        
        ax[1].set_title("Cropped")
        ax[1].imshow(cropped_img)
        
    return cropped_img