import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CVDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform_input=None, transform_target=None):
        """
        Args:
            input_dir (str): Directory with input images (e.g., low-res images).
            target_dir (str): Directory with target images (e.g., high-res images or segmented masks).
            transform_input (callable, optional): Optional transform to be applied on the input images.
            transform_target (callable, optional): Optional transform to be applied on the target images.
        """
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform_input = transform_input
        self.transform_target = transform_target

        # Assume input and target directories have corresponding file names
        self.input_images = sorted(os.listdir(self.input_dir))
        self.target_images = sorted(os.listdir(self.target_dir))
        
        assert len(self.input_images) == len(self.target_images), "Input and target datasets should have the same number of images."

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        """
        Fetches the input image and the corresponding target image.
        """
        input_img_path = os.path.join(self.input_dir, self.input_images[idx])
        target_img_path = os.path.join(self.target_dir, self.target_images[idx])

        # Load the images
        input_image = Image.open(input_img_path).convert('RGB')
        target_image = Image.open(target_img_path).convert('RGB')

        # Apply transformations if provided
        if self.transform_input:
            input_image = self.transform_input(input_image)
        if self.transform_target:
            target_image = self.transform_target(target_image)

        return input_image, target_image


def get_image2image_dataloaders(batch_size=32, shuffle=True, num_workers=0, input_dir=None, target_dir=None, transform_input=None, transform_target=None):
    """
    A utility function to create DataLoader instances for image-to-image tasks.
    
    Args:
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses to use for data loading.
        input_dir (str): Path to the directory containing input images.
        target_dir (str): Path to the directory containing target images.
        transform_input (torchvision.transforms): Transformations to be applied to input images.
        transform_target (torchvision.transforms): Transformations to be applied to target images.
        
    Returns:
        DataLoader: DataLoader object for the image-to-image dataset.
    """
    # Define default transformations for both input and target images if not provided
    transform_input = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_target = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Instantiate the dataset
    dataset = CVDataset(input_dir=input_dir, target_dir=target_dir, 
                                 transform_input=transform_input, 
                                 transform_target=transform_target)
    
    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader


if __name__ == "__main__":
    rootdir = r'C:\Lucas\Project2024-2025\Code'
    datadir = os.path.join(rootdir, 'dataset1')
    # Example usage:
    train_loader = get_image2image_dataloaders(batch_size=16, shuffle=True, 
                                               input_dir=os.path.join(datadir, 'input', 'train_del_bad'), 
                                               target_dir=os.path.join(datadir, 'target', 'train_del_bad'))

    # Iterate over data
    for input_images, target_images in train_loader:
        print(f"Input batch shape: {input_images.shape}, Target batch shape: {target_images.shape}")
