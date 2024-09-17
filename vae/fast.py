from PIL import Image
import numpy as np
import pandas as pd
import sys, os, csv
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch


#################### Setup
# Input
dim = 17

# Matrix (11x3x3)
m_match = np.array([i*0.1 for i in range(0, 11)])
m = np.array(
    
    [
        [
            [1.000000,	0.000000,	-0.000000],
            [0.000000,	1.000000,	0.000000], 
            [-0.000000,	-0.000000,	1.000000],
        ],
        [
            [0.866435,	0.177704,	-0.044139],
[0.049567,	0.939063,	0.011370],
[-0.003453,	0.007233,	0.996220],
        ],
        [[0.760729,	0.319078,	-0.079807],
[0.090568,	0.889315,	0.020117],
[-0.006027,	0.013325,	0.992702],],
        [[0.675425,	0.433850,	-0.109275],
[0.125303,	0.847755,	0.026942],
[-0.007950,	0.018572,	0.989378]	,],
        [[0.605511,	0.528560,	-0.134071],
[0.155318,	0.812366,	0.032316],
[-0.009376,	0.023176,	0.986200],],
        [[0.547494,	0.607765,	-0.155259],
[0.181692,	0.781742,	0.036566],
[-0.010410,	0.027275,	0.983136],],
        [[0.498864,	0.674741,	-0.173604],
[0.205199,	0.754872,	0.039929],
[-0.011131,	0.030969,	0.980162],],
        [[0.457771,	0.731899,	-0.189670],
[0.226409,	0.731012,	0.042579],
[-0.011595,	0.034333,	0.977261],],
        [[0.422823,	0.781057,	-0.203881],
[0.245752,	0.709602,	0.044646],
[-0.011843,	0.037423,	0.974421],],
        [[0.392952,	0.823610,	-0.216562],
[0.263559,	0.690210,	0.046232],
[-0.011910,	0.040281,	0.971630],],
        [[0.367322,	0.860646,	-0.227968],
[0.280085,	0.672501,	0.047413],
[-0.011820,	0.042940,	0.968881],]
    ]
)

# Functions
def generate_3dlut(matrix: np.array) -> np.array:
        if matrix.shape != (3, 3):
            raise ValueError("Matrix must be 3x3")

        grid = np.linspace(0, 1, dim)
        lut = np.zeros((dim, dim, dim, 3))

        for r in range(dim):
            for g in range(dim):
                for b in range(dim):
                    rgb = np.array([grid[r], grid[g], grid[b]])
                    transformed_rgb = np.dot(matrix, rgb)
                    #transformed_rgb = np.clip(transformed_rgb, 0, 1)
                    lut[r, g, b] = transformed_rgb

        return lut

def write_lut(lut: np.array, lut_path: str) -> None:
    lut_dim = lut.shape[0]
    
    with open(lut_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['R', 'G', 'B', 'R_out', 'G_out', 'B_out'])  # Header
        
        for r in range(lut_dim):
            for g in range(lut_dim):
                for b in range(lut_dim):
                    r_in, g_in, b_in = r / (lut_dim - 1), g / (lut_dim - 1), b / (lut_dim - 1)
                    r_out, g_out, b_out = lut[r, g, b]
                    writer.writerow([r_in, g_in, b_in, r_out, g_out, b_out])

def read_lut(filename: str) -> torch.tensor:
    df = pd.read_csv(filename)
    lut = np.zeros((dim, dim, dim, 3)) # R x G x B x Output
    
    for _, row in df.iterrows():
        r, g, b = int(row['R'] * (dim - 1)), int(row['G'] * (dim - 1)), int(row['B'] * (dim - 1))
        lut[r, g, b] = row[['R_out', 'G_out', 'B_out']]
    
    lut = torch.tensor(lut, dtype=torch.float32)
    
    lut = lut.permute(3, 2, 1, 0)[None] # N x Output x R x G x B
    return lut

#################### LUT 

def get_lut_fn(lvl: float):
    if lvl == 0:
        return 'lut0-0.txt'
    if lvl == 1:
        return 'lut1-0.txt'
    return f"lut{lvl}".replace(".", '-')+'.txt'


# Create LUT
def make_lut(lvl: float, lutdir: str):
    # Define LUT Path
    lut_path = os.path.join(lutdir, get_lut_fn(lvl))
    m_interp = np.array([
        [np.interp(lvl, m_match, m[:,0,0]), np.interp(lvl, m_match, m[:,0,1]), np.interp(lvl, m_match, m[:,0,2])],
        [np.interp(lvl, m_match, m[:,1,0]), np.interp(lvl, m_match, m[:,1,1]), np.interp(lvl, m_match, m[:,1,2])],
        [np.interp(lvl, m_match, m[:,2,0]), np.interp(lvl, m_match, m[:,2,1]), np.interp(lvl, m_match, m[:,2,2])]
    ])
    lut = generate_3dlut(m_interp)
    write_lut(lut, lut_path)


#################### Image

def make_image(lvl: float, image: torch.Tensor, lutdir: str) -> torch.Tensor: # image in PIL RGB format: Image.open(fp).convert('RGB')
    # Read LUT
    lut_path = os.path.join(lutdir, get_lut_fn(lvl))
    lut = read_lut(lut_path)

    # Open Image
    image = image * 2 - 1

    # Format dimensions CxHxW -> NxDxHxWxC
    image = torch.tensor(image, dtype=torch.float32)[None, None]
    image = image.permute(0, 1, 3, 4, 2)

    # Apply LUT
    res = F.grid_sample(lut, image, padding_mode='border', align_corners=True) # align_corners is CRUCIAL
    res = res.permute(0, 2, 1, 3, 4) # NxCxDxHxW -> NxDxCxHxW
    res = res.squeeze() # CxHxW
    res = res.clamp(0, 1)

    # Save Transform: 
    return res

