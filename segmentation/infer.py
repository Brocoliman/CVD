import torch
from config import *
from utils import load_checkpoint, save_preds_dual, get_loaders
from segmentation.model_UNet import UNET

MODEL_NAME = "segmentation/saved_models/ade20k_vB_1.tar"


model = UNET(in_channels=3, out_channels=CLASSES+1).to(DEVICE)
load_checkpoint(torch.load(MODEL_NAME), model)

train_loader, val_loader = get_loaders(
        CLASSES,
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        blank_transform,
        NUM_WORKERS,
        PIN_MEMORY,
        FN_FORMAT
    )

save_preds_dual(val_loader, model)