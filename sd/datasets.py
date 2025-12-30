import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt 

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # List all image files in the directory
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Returning a dictionary to match your trainer's 'batch['image']' logic
        return {'image': image}

def show_images(images, title = 'Images'):
    images = images.detach().cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    images = np.clip(images, 0, 1)
    fig, axes = plt.subplots(1, len(images), figsize = (15, 5))
    for i, img in enumerate(images):
        axes[i].imshow(fig)
        axes[i].axis('off')
    plt.subtitle('title')
    plt.show()
