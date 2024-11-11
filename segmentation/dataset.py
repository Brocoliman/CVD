import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, fn_format=(".jpg", "_mask.gif"), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.fn_format = fn_format
        self.transform = transform
        self.imagelist = os.listdir(image_dir)

    def __len__(self):
        return len(self.imagelist)
    
    def __getitem__(self, index):
        imgpath = os.path.join(self.image_dir, self.imagelist[index])
        maskpath = os.path.join(self.mask_dir, self.imagelist[index].replace(*self.fn_format))
        images = np.array(Image.open(imgpath).convert("RGB"))
        masks = np.array(Image.open(maskpath).convert("L"), dtype=np.float32) # grayscale 
        masks[masks==255.0] = 1.0 # normalize white values for sigmoid

        if self.transform is not None:
            augmentations = self.transform(image=images, mask=masks)
            images = augmentations["image"]
            masks = augmentations["mask"]
        
        return images, masks