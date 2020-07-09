import os
import pandas as pd
from skimage import io
from csv import reader

BBBC038 = "/raid/data/BBBC038/"
training_imgs_dir = "/raid/data/BBBC038/training/"


def load_imgs_masks():
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

def load_COLOR_data(csv_lines=csv_lines):
    def_png_list = []
    def_img_paths = []
    def_mask_colls = []
    
    bw_png_list = []
    bw_img_paths = []
    bw_mask_colls = []
    
    purple_png_list = []
    purple_img_paths = []
    purple_mask_colls = []
    
    pink_png_list = []
    pink_img_paths = []
    pink_mask_colls = []
    
    for row in csv_lines: # make this a parameter in the function call
        if "Default" in row:
            def_png_list.append(row[1])
        elif "TissueBW" in row:
            bw_png_list.append(row[1])
        elif "Purple" in row:
            purple_png_list.append(row[1])
        elif "Pink-Purple" in row:
            pink_png_list.append(row[1])
            
    del def_png_list[0] # first entry in the csv file is [Image, Type]

    # make these generic and independent of selected color
    for png in def_png_list:
        img_path = training_imgs_dir + png[:-4] + "/images/" + png
        def_img_paths.append(img_path)
        
        mask_path = training_imgs_dir + png[:-4] + "/masks/*.png"
        imcoll = io.collection.ImageCollection(mask_path)
        def_mask_colls.append(imcoll)
        
    for png in bw_png_list:
        img_path = training_imgs_dir + png[:-4] + "/images/" + png
        bw_img_paths.append(img_path)
        
        mask_path = training_imgs_dir + png[:-4] + "/masks/*.png"
        imcoll = io.collection.ImageCollection(mask_path)
        bw_mask_colls.append(imcoll)
        
    for png in purple_png_list:
        img_path = training_imgs_dir + png[:-4] + "/images/" + png
        purple_img_paths.append(img_path)
        
        mask_path = training_imgs_dir + png[:-4] + "/masks/*.png"
        imcoll = io.collection.ImageCollection(mask_path)
        purple_mask_colls.append(imcoll)
        
    for png in pink_png_list:
        img_path = training_imgs_dir + png[:-4] + "/images/" + png
        pink_img_paths.append(img_path)
        
        mask_path = training_imgs_dir + png[:-4] + "/masks/*.png"
        imcoll = io.collection.ImageCollection(mask_path)
        pink_mask_colls.append(imcoll)
    
    all_pngs = [def_png_list, bw_png_list, purple_png_list, pink_png_list]
    all_img_paths = [def_img_paths, bw_img_paths, purple_img_paths, pink_img_paths]
    all_mask_colls = [def_mask_colls, bw_mask_colls, purple_mask_colls, pink_mask_colls]
    
    return all_pngs, all_img_paths, all_mask_colls
