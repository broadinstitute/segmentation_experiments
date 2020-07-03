from skimage import io
import numpy as np
import matplotlib.pyplot as plt


def random_crop(img_path, crop_size=64, loud=False):
    img = io.imread(img_path)[:, :, :3]                         # neglect alpha value
    row_start = np.random.randint(0, img.shape[0] - crop_size)  # select a random row and column
    col_start = np.random.randint(0, img.shape[1] - 64)         # from origin while allowing for a full crop

    row_end = row_start + 64
    col_end = col_start + 64                            
    
    cropped_img = img[row_start:row_end, col_start:col_end, :]    # slice the image along our desired border
    
    if loud:
        print("Original image size is %d, %d. Crop starts at %d,%d. Cropped image is %dx%d pixels" 
              %(img.shape[0], img.shape[1], row_start, col_start, crop_size, crop_size))
        
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,8))
        
        ax[0].set_title("Original")
        ax[0].imshow(img)
        
        ax[1].set_title("Cropped")
        ax[1].imshow(cropped_img)
        
    return cropped_img
