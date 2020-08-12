from skimage.transform import resize
from functions.sizes import compute_avg_size
import matplotlib.pyplot as plt

def rescale_img_comp(image, composite, target_size, loud=False):
    """
    Function which takes image, ground truth, and target size
    and returns a new image with a new composite, each of which
    is rescaled to a new size that approximates the requested 
    target size.
    
    """
    curr_size = compute_avg_size(composite)
    ratio = target_size / curr_size
    ratio = ratio ** (1/2)
    new_shape = (int(image.shape[0] * ratio),
                 int(image.shape[1] * ratio))
    
    resized_img = resize(image, new_shape, anti_aliasing=True)
    resized_comp = resize(composite, new_shape, anti_aliasing=False,
                            preserve_range=True, order=0)
    
    if loud:
        plt.imshow(resized_img); plt.figure()
        plt.imshow(resized_comp)
        new_size = compute_avg_size(resized_comp)
        print("Original size: %f, Rescaled size: %f" %(curr_size, new_size))
        
    return resized_img, resized_comp
        