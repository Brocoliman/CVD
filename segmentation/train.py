import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from config import *
from segmentation.model_UNet import UNET
from utils import (
    load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_preds_dual
)
torch.cuda.empty_cache()

# Code
def train(loader, model, optimizer, loss_fn, scaler): # does one epoch
    loop = tqdm(loader)

    for batch_idx, (data, targets, _) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            preds = model(data)
            loss = loss_fn(preds,targets)
            """if batch_idx % 100 == 0:
                print(torch.argmax(preds, 1))
                print(targets)
"""
        # back
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    # Model setup
    model = UNET(in_channels=3, out_channels=CLASSES+1).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    # Load Data
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

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint, epoch=epoch, filename=MODEL_NAME)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # save images
        save_preds_dual(val_loader, model, folder='segmentation/saved_images', device=DEVICE)

if __name__ == "__main__":
    main()