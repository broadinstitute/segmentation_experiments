import os
import pandas as pd
from skimage import io
from csv import reader

BBBC038 = "/raid/data/BBBC038/"
training_imgs_dir = "/raid/data/BBBC038/training/"


def load_imgs_masks():
    """
    Function which loads ALL images and mask collections (unorganized)
    
    Returns 4 arrays
        First contains paths to all images
        Second contains names of each image
        Third contains names of all masks for each image
        Fourth contains collections of all masks corresponding to each image
    """
    img_paths = []
    img_names = []
    mask_png_list = []
    mask_colls = []

    for img_name in os.listdir(training_imgs_dir):
        path_image = f"{training_imgs_dir}/{img_name}/images/{img_name}.png" # path of each image png
        path_masks = f"{training_imgs_dir}/{img_name}/masks/*.png"           # path to each directory of masks for each image

        mask_coll = io.collection.ImageCollection(path_masks)     # image collection storing all masks for given image
        mask_dir = f"{training_imgs_dir}/{img_name}/masks/"       
        mask_dir = os.listdir(mask_dir)                           # list of all masks for an image

        img_names.append(img_name)                                # store list of image names
        img_paths.append(path_image)                              # store paths to each image
        mask_png_list.append(mask_dir)                            # store lists of mask names
        mask_colls.append(mask_coll)                              # store mask collections

    return img_paths, img_names, mask_png_list, mask_colls

def load_labeled_data():
    """
    Function to load 'labeled' data
    
    Returns image groups, table entries, and csv rows
    """
    table_entries = []
    image_groups = pd.read_csv(BBBC038 + "training_classifications.csv")
    
    for i, entry in image_groups.iterrows():
        table_entries.append(entry)             # store each line in csv table in array 
        
    csv_lines = []
    with open(BBBC038 + "training_classifications.csv") as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            csv_lines.append(row)               # store arrays of [classification, image names]
            
    # example
    # print(csv_lines[1])
    #   ['SuperBig', 'a102535b0e88374bea4a1cfd9ee7cb3822ff54f4ab2a9845d428ec22f9ee2288.png']
        
    
    return image_groups, table_entries, csv_lines

image_groups, table_entries, csv_lines = load_labeled_data()

def load_COLOR_data(color="Default", csv_lines=csv_lines):
    """
    Function to load all images of the specified color.
    
    Options include "Default", "Pink-Purple", "TissueBW", "Purple"
    
    returns 3 arrays. 
        First array has the name of the png 
        Second is the path to the image
        Third is the mask collection corresponding to the image
    """
    if (color != "Default") and (color != "Pink-Purple") and (color != "TissueBW") and (color != "Purple"):
        raise ValueError("Must select a supported color. These include 'Default', 'Pink-Purple', 'TissueBW', 'Purple'.")
    
    png_list = []
    img_paths = []
    mask_colls = []

    for row in csv_lines:
        if color in row:
            png_list.append(row[1])
            
    if color == "Default":
        del png_list[0]   # Because csv file begins with "Default", first entry is [Image, Type]
        
    for png in png_list:
        path = training_imgs_dir + png[:-4] + "/images/" + png
        img_paths.append(path)
        
        mask_path = training_imgs_dir + png[:-4] + "/masks/*.png"
        imcoll = io.collection.ImageCollection(mask_path)
        mask_colls.append(imcoll)
        
    return png_list, img_paths, mask_colls
