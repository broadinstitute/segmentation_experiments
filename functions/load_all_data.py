import os
import pandas as pd
from skimage import io
from csv import reader

BBBC038 = "/raid/data/BBBC038/"
training_imgs_dir = "/raid/data/BBBC038/training/"

def load_labeled_data():
    """
    Updated function to load labelled data
    
    Incorporates updated labels to CSV files in /raid/data/BBBC038/fix_training_classifications.csv
    
    returns image groups, table entries, and csv rows
    """
    table_entries = []
    image_groups = pd.read_csv(BBBC038 + "fix_training_classifications.csv")
    
    for i, entry in image_groups.iterrows():
        table_entries.append(entry)             # store each line in csv table in array 
        
    csv_lines = []
    with open(BBBC038 + "fix_training_classifications.csv") as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            csv_lines.append(row)               # store arrays of [classification, image names]
            
    # example
    # print(csv_lines[1])
    #   ['0', 'a102535b0e88374bea4a1cfd9ee7cb3822ff54f4ab2a9845d428ec22f9ee2288.png', 'Default', 'SuperBig']
        
    
    return image_groups, table_entries, csv_lines

def load_imgs_masks():
    """
    Function which loads ALL images and mask collections (unorganized)
    
    Returns 3 arrays
        First contains image objects (as numpy arrays)
        Second contains collection of masks for each image
        Third contains paths to all images
    """
    img_objs = []
    img_paths = []
    mask_colls = []

    for img_name in os.listdir(training_imgs_dir):
        path_image = f"{training_imgs_dir}/{img_name}/images/{img_name}.png" # path of each image png
        path_masks = f"{training_imgs_dir}/{img_name}/masks/*.png"           # path to each directory of masks for each image
        
        img = io.imread(path_image)                               # load image objects into array

        mask_coll = io.collection.ImageCollection(path_masks)     # image collection storing all masks for given image

        img_paths.append(path_image)                              # store paths to each image
        mask_colls.append(mask_coll)                              # store mask collections
        img_objs.append(img)

    return img_objs, mask_colls, img_paths

csv_lines = load_labeled_data()[2]

def load_data_by_color(color="Default", csv_lines=csv_lines):
    """
    Function to load all images of the specified color.
    
    Options include "Default", "Pink-Purple", "Gray-Scales", "Purple"
    
    returns 3 arrays. 
        First returns image objects 
        Second returns the paths to the images
        Third returns the mask collections corresponding to each image
    """
    if (color != "Default") and (color != "Pink-Purple") and (color != "Gray-Scales") and (color != "Purple"):
        raise ValueError("Must select a supported color. These include 'Default', 'Pink-Purple', 'Gray-Scales', 'Purple'.")
    
    png_list = []       # needed to find image paths; do not return
    
    image_objects = []
    img_paths = []
    mask_colls = []

    for row in csv_lines:
        if color in row:
            png_list.append(row[1])
            
    for png in png_list:
        path = training_imgs_dir + png[:-4] + "/images/" + png
        img_paths.append(path)
        
        img = io.imread(img_paths[-1])
        image_objects.append(img)
        
        mask_path = training_imgs_dir + png[:-4] + "/masks/*.png"
        imcoll = io.collection.ImageCollection(mask_path)
        mask_colls.append(imcoll)
        
    return image_objects, mask_colls, img_paths
