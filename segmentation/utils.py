import torch
import torchvision
from dataset import SegDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, epoch, filename='segmentation/saved_models/checkpoint.tar'):
    print("[Action] Saving Checkpoint")
    torch.save(state, filename[:-4] + str(epoch) + filename[-4:])

def load_checkpoint(checkpoint, model):
    print("[Action] Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform, num_workers, pin_memory):
    train_ds = SegDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        shuffle=True
    )
    val_ds = SegDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        shuffle=False
    )
    return train_loader, val_loader

def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    print(f"[Event] P2P: {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"[Event] Dice: score {dice_score / len(loader)}")

    model.train()

def save_preds(loader, model, folder='segmentation/saved_images', device='cuda'):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/y{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/gt{idx}.png")
        if idx > 20: break
