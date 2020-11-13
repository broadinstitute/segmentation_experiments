import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from functions.load_all_data import load_data_by_color, load_imgs_masks
from functions.composites import composite_masks
from functions.crop_image import random_crop
from functions.rescaling import rescale_img_comp
from functions.sizes import compute_avg_size

class ColorDataSet:
    """
    Data structure containing randomly cropped images
    for training a neural network identifying the color
    of an images.
    """
    
    def __init__(self, num_samples, crop_size=64):
        self.samples = []
        self.labels = []
    
        default_imgs = load_data_by_color("Default")[0]
        purple_imgs = load_data_by_color("Purple")[0]
        gray_imgs = load_data_by_color("Gray-Scales")[0]
        pink_imgs = load_data_by_color("Pink-Purple")[0]
        
        # works similarly to training_samples function in load_training_data.py

        for i in range(num_samples // 4):
            def_index = np.random.randint(0, len(default_imgs))
            default_img = default_imgs[def_index]
            cropped_default_img = random_crop(default_img, crop_size=crop_size)[0]
            self.samples.append(cropped_default_img)
            self.labels.append(0.)
            
            purp_index = np.random.randint(0, len(purple_imgs))
            purple_img = purple_imgs[purp_index]
            cropped_purple_img = random_crop(purple_img, crop_size=crop_size)[0]
            self.samples.append(cropped_purple_img)
            self.labels.append(1.)
            
            gray_index = np.random.randint(0, len(gray_imgs))
            gray_img = gray_imgs[gray_index]
            cropped_gray_img = random_crop(gray_img, crop_size=crop_size)[0]
            self.samples.append(cropped_gray_img)
            self.labels.append(2.)
            
            pink_index = np.random.randint(0, len(pink_imgs))
            pink_img = pink_imgs[pink_index]
            cropped_pink_img = random_crop(pink_img, crop_size=crop_size)[0]
            self.samples.append(cropped_pink_img)
            self.labels.append(3.)
            
        self.ToTensor()
            
    def __getitem__(self, idx):
        # select one item from 
        # potentially return dictionary with the 2 items
        # see if this is necessary!
        return self.samples[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.samples)
    
    def ToTensor(self):
        """
        swap color axis because
        numpy image: H x W x C
        torch image: C X H X W
        """
        for i in range(len(self.samples)):
            self.samples[i] = self.samples[i].transpose((2,0,1))
            self.samples[i] = torch.tensor(self.samples[i].astype(float) / 255.).type("torch.FloatTensor")
            
            
        self.labels = torch.LongTensor(self.labels)
            
    