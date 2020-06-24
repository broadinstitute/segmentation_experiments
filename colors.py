def unmix_purple_img(purple_img_path, loud=False):
    img = io.imread(purple_img_path)                                  # loads image from path
    
    hematoxylin_matrix = np.ones((3,3)) * (0.644, 0.717, 0.267)       # cell profiler matrix for purple images
    stain_img = img[:, :, [0, 1, 2]]                                  # need only first 3 channels to separate stains
    separated_img = separate_stains(stain_img, hematoxylin_matrix)    # apply stain matrix to image
    
    if loud:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,8))
        
        ax[0].set_title("Original")
        ax[0].imshow(img)
        
        ax[1].set_title("Hematoxylin")
        ax[1].imshow(separated_img[:, :, 0])
    
    return separated_img[:, :, 0]

def unmix_pink_imgs(pink_img_path, loud=False):
    img = io.imread(pink_img_path)
    
    stain_img = img[:, :, [0, 1, 2]]
    separated_img = separate_stains(stain_img, rbd_from_rgb)
    
    if loud:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,8))
        
        ax[0].set_title("Original")
        ax[0].imshow(img)
        
        ax[1].set_title("RBD")
        ax[1].imshow(separated_img[:, :, 1])
        
    return separated_img[:, :, 1]

unmixed_pink = unmix_pink_imgs(pink_img_paths[3], True)

