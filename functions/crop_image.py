from skimage import io
import numpy as np
import matplotlib.pyplot as plt


def random_crop(img_path, crop_size=64, loud=False):
    img = io.imread(img_path)[:, :, :3]      # neglect alpha value
    row_max = img.shape[0] - crop_size       # get the maximum row or column we can begin cropping
    col_max = img.shape[1] - crop_size
    
    random_start = np.random.randint(0, min(row_max, col_max))  # select a random point to start cropping within our boundary
    end = random_start + crop_size                              
    
    cropped_img = img[random_start:end, random_start:end, :]    # slice the image along our desired border
    
    if loud:
        print("Original image size is %d, %d. Crop starts at %d,%d. Cropped image is %dx%d pixels" 
              %(img.shape[0], img.shape[1], random_start,random_start, crop_size, crop_size))
        
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,8))
        
        ax[0].set_title("Original")
        ax[0].imshow(img)
        
        ax[1].set_title("Cropped")
        ax[1].imshow(cropped_img)
        
    return cropped_img
