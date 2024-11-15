import torch
import torchvision
import torchvision.transforms as transforms
from dataset import SegDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, epoch=0, filename='segmentation/saved_models/checkpoint.tar'):
    print("[Action] Saving Checkpoint...")
    if epoch is not None:
        torch.save(state, filename[:-4] + str(epoch) + filename[-4:])
    else:
        torch.save(state, filename)
    print("[Event] Checkpoint Saved.")

def load_checkpoint(checkpoint, model):
    print("[Action] Loading Checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])
    print("[Event] Checkpoint Loaded.")

def get_loaders(classes, 
                train_dir, train_maskdir, val_dir, val_maskdir, 
                batch_size, train_transform, val_transform, mask_transform,
                num_workers, pin_memory, 
                fn_format, val_gt=True):
    train_ds = SegDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        classes=classes,
        fn_format=fn_format,
        transform=train_transform,
        mask_transform=mask_transform,
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
        mask_dir=val_maskdir if val_gt else None,
        classes=classes,
        fn_format=fn_format,
        transform=val_transform,
        mask_transform=mask_transform,
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

def save_preds_dual(loader, model, limit=20, folder='segmentation/saved_images', device='cuda'):
    model.eval()
    toPIL = transforms.ToPILImage(mode='L')
    for idx, (x, y, fn) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
        torchvision.utils.save_image(torch.argmax(preds, 1).unsqueeze(1).float(), f"{folder}/gt{idx}.png", nrow=4, padding=2, normalize=True)
        torchvision.utils.save_image(y.unsqueeze(1).float(), f"{folder}/y{idx}.png", nrow=4, padding=2, normalize=True)

        if limit is not None and idx > limit: break

def save_preds_mono(loader, model, limit=None, folder='segmentation/saved_images', device='cuda'):
    model.eval()
    for idx, x in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/y{idx}.png")
        if limit is not None and idx > limit: break
