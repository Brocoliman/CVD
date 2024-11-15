import os
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, classes=2, fn_format=None, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.classes = classes
        self.fn_format = fn_format
        self.transform = transform
        self.mask_transform = mask_transform
        self.imagelist = os.listdir(image_dir)

    def __len__(self):
        return len(self.imagelist)
    
    def __getitem__(self, index):
        imgpath = os.path.join(self.image_dir, self.imagelist[index])
        images = np.array(Image.open(imgpath).convert("RGB"))
        if self.mask_dir is not None:
            maskpath = os.path.join(self.mask_dir, self.imagelist[index] if self.fn_format is None else self.imagelist[index].replace(*self.fn_format))
            masks = np.array(Image.open(maskpath).convert("L")) # grayscale 

        if self.transform is not None:
            augmentations = self.transform(image=images)
            images = augmentations["image"]
        if self.mask_transform is not None:
            maskt = self.mask_transform(image=masks)
            masks = maskt["image"]

        masks = masks.to(torch.long).squeeze()
        
        return (images, masks, imgpath[-12:-4]) if self.mask_dir is not None else images