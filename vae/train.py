import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # For progress bar
from model import CVDVAE
from dataset import get_image2image_dataloaders

# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda', save_path=''):
    model.to(device)
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    
    # Over epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Train & Validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            total_samples = 0

            # Iterate over data
            for inputs, targets in tqdm(dataloaders[phase]):
                inputs, targets = inputs.to(device), targets.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, _, _ = model(inputs)
                    loss = criterion(outputs, targets)

                    # Backward pass and optimization only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                total_samples += targets.size(0)

            epoch_loss = running_loss / total_samples

            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            # Deep copy the model if it has the best loss so far
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
            
            # Save a copy at this epoch
            if phase == 'train':
                model = model.cpu()
                torch.save({
                    'epoch': epoch, 
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss
                    }, 
                    os.path.join(save_path, 'checkpoint', f'model{epoch}.pth')
                    )
                torch.save(model.state_dict(), os.path.join(save_path, 'infer', f'model{epoch}.pth'))
                model = model.cuda()

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    
    rootdir = r'C:\Lucas\Project2024-2025\Code'
    datadir = os.path.join(rootdir, 'dataset1')
    savedir = os.path.join(rootdir, 'saves')

    # Hyperparameters
    num_epochs = 25
    batch_size = 32
    learning_rate = 0.001

    # Instantiate model, criterion (loss function), and optimizer
    model = CVDVAE()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader
    train_loader = get_image2image_dataloaders(batch_size=16, shuffle=True, 
                                               input_dir=os.path.join(datadir, 'input', 'train_del_bad'), 
                                               target_dir=os.path.join(datadir, 'target', 'train_del_bad'))
    val_loader = get_image2image_dataloaders(batch_size=16, shuffle=True, 
                                               input_dir=os.path.join(datadir, 'input', 'test'), 
                                               target_dir=os.path.join(datadir, 'target', 'test'))

    # Dictionary to hold the DataLoaders
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    if not os.path.isdir(os.path.join('saves', 'infer')):
        os.mkdir(os.path.join('saves', 'infer'))
    if not os.path.isdir(os.path.join('saves', 'checkpoint')):
        os.mkdir(os.path.join('saves', 'checkpoint'))

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, save_path=savedir)

    # Save the model
    torch.save(model.state_dict(), 'best_model.pth')
