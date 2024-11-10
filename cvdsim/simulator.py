import argparse
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
from fast import make_image, change_lvl

root = r"/home/brocolimanx/Desktop/Image-Adaptive-3DLUT-master"

sys.path.append(root)

from image_adaptive_lut_evaluation import generator_wrapper

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=563, help="epoch to load the saved checkpoint")
parser.add_argument("--model", type=str, default="sRGB_Jan4", help="directory of saved models")
parser.add_argument("--model_dir", type=str, default="saved_models_backup/%s", help="path to save model")
parser.add_argument("--library", type=str, default=os.path.join(root, 'data', 'easy_eval'), help="images to test")
opt = parser.parse_args()

############################## Modals ###############################

"""
Displays several modals:
Original - *
Daltonized - *: based on select algorithm
Groundtruth - *: based on groundtruth file path

"""

class ModalTable:
    def __init__(self, model_functions: list[any] = [], 
                 model_args: list[list[any]] = [], 
                 model_kwargs: list[dict[any]] = [], 
                 model_names: list[str] = [], 
                 **gen_table):
        """
        constructs flow using modals as required by user
        gen_table requires:
        - show
        - show_sim
        - show_daltsim
        - show_gt
        - gt_path
        """
        self.mf = model_functions
        self.margs = model_args
        self.mkwargs = model_kwargs
        self.mn = model_names
        assert len(self.mf) == len(self.margs) == len(self.mkwargs) == len(self.mn)
        self.num = len(self.mf)
        self.gen_table = gen_table

    def _tensor(self, input):
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(Image.fromarray(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))).cuda()

    def _cv(self, input):
        transform = transforms.Compose([transforms.ToPILImage()])
        return cv2.cvtColor(np.array(transform(input.detach())), cv2.COLOR_RGB2BGR)

    def run(self, img: Image, lvl: float = 1):
        t_img = self._tensor(img)
        t_img = t_img.cuda()
        t_img = t_img[None]

        dalt = [self.mf[i](t_img, *self.margs[i], **self.mkwargs[i]) for i in range(self.num)]
        sim = make_image(t_img) if self.gen_table['show_sim'] else None
        daltsim = [make_image(dalt[i]) for i in range(self.num)] if self.gen_table['show_daltsim'] else [None for _ in range(self.num)]

        cno = [t_img] + dalt
        cdo = [sim] + daltsim

        if self.gen_table['show']:
            cv2.imshow("Original", self._cv(t_img.squeeze()))
            cv2.imshow("Simulated", self._cv(sim))
            for i in range(0, self.num):
                cv2.imshow(self.mn[i], self._cv(dalt[i].squeeze()))
                cv2.imshow("Sim "+self.mn[i], self._cv(daltsim[i]))
        
        return cno, cdo

class Carousel:
    def __init__(self, library_dir: str, modaltable: ModalTable = ModalTable(model_functions=[],
               model_args=[],
               model_kwargs=[],
               model_names=[],
               show=True, show_sim=True, show_daltsim=True)):
        self.library_dir = library_dir
        self.library = os.listdir(library_dir)
        self.num_images = len(self.library)
        self.modaltable = modaltable
        self.lvl = 1
        change_lvl(1)
    
    def run(self):
        idx = 0
        while True:
            # Input from camera
            frame = cv2.imread(os.path.join(self.library_dir, self.library[idx]))
            self.modaltable.run(frame)

            ##### Wait Key #####
            key_down = chr(cv2.waitKey(1) & 0xFF)
            if key_down == 'w': idx -= 1
            if key_down == 'e': idx += 1
            if key_down in ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']: 
                if key_down == '`': self.lvl = 0
                elif key_down == '0': self.lvl = 1
                else: self.lvl = int(key_down)/10
                change_lvl(self.lvl)
                print(f"[EVENT] Changed CVD Severity to {self.lvl}")
                
            if key_down == 'q': break
            idx %= self.num_images

        cv2.destroyAllWindows()

############################## Evaluation ###############################
if __name__ == "__main__":
    def apply_lut(img, LUT, repeat=False):
        # scale im between -1 and 1 since its used as grid input in grid_sample
        img = (img - .5) * 2.
        # grid_sample expects NxDxHxWx3 (1x1xHxWx3)
        img = img.permute(0, 2, 3, 1)[:, None]
        # add batch dim to LUT
        if repeat:
            LUT = LUT[None].repeat(img.shape[0],1,1,1,1)
        else:
            LUT = LUT[None]
        # apply LUT
        result = F.grid_sample(LUT, img, mode='bilinear', padding_mode='border', align_corners=True)
        # drop added dimensions and permute back
        result = result[:, :, 0]
        #self.LUT, output = self.TrilinearInterpolation(self.LUT, x)
        return result
    def hr_lut(img, dim=17):
        lines = open('lut3d_rgb.txt').readlines()
        buffer = np.zeros((3,dim,dim,dim), dtype=np.float32)
        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    n = i * dim*dim + j * dim + k
                    x = lines[n].split()
                    buffer[0,i,j,k] = float(x[0]) / 255.
                    buffer[1,i,j,k] = float(x[1]) / 255.
                    buffer[2,i,j,k] = float(x[2]) / 255.
        LUT = nn.Parameter(torch.from_numpy(buffer)).cuda()
        result = apply_lut(img, LUT)
        return result
    m = ModalTable(model_functions=[generator_wrapper, generator_wrapper, hr_lut],
                model_args=[list(), list(), list()],
                model_kwargs=[{'bin_class':True},{'bin_class':False}, dict()],
                model_names=['[DaltNET]', '[DaltNET-raw]', '[HR]'],
                show=True, show_sim=True, show_daltsim=True)
    c = Carousel(opt.library, m)
    c.run()

"""
python camera.py --epoch 505 --model_dir data/saved_models/%s --model sRGB_411

10/4:
python cvdsim/camera.py --epoch 399 --model_dir data/saved_models/%s --model 103

10/15:
python cvdsim/camera.py --epoch 399 --model_dir data/saved_models/%s --model 1015

"""