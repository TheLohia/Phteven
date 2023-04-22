import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Define the transforms for the training data
if __name__ == "__main__": 
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class MeatDataset(Dataset):
    """A PyTorch Dataset class for loading images from a directory containing meat freshness images.
    
    Args:
        data_dir (str): The path to the directory containing the image files.
        transform (callable, optional): Optional transforms to be applied to the images.
    """
    def __init__(self, data_dir, transform=None):
      """
        Initializes a new instance of the MeatDataset class.
        
        Args:
            data_dir (str): The path to the directory containing the image files.
            transform (callable, optional): Optional transforms to be applied to the images.
            """
      self.data_dir = data_dir
      self.file_names = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
      self.transform = transform

    def __len__(self):
      """
        Returns the number of images in the dataset.
        """
      return len(self.file_names)

    def __getitem__(self, idx):
       """
        Returns the image and corresponding label at the given index in the dataset.
        
        Args:
            idx (int): The index of the image to retrieve.
            
        Returns:
            tuple: A tuple containing the image and corresponding label.
        """
       file_name = self.file_names[idx]
       img_path = os.path.join(self.data_dir, file_name)
       img_class = file_name.split("-")[0]

       # Load the image
       img = Image.open(img_path)
       # Apply the transforms
       if self.transform:
        img = self.transform(img)

        # Convert the class label to a tensor
        label = torch.tensor([0, 0, 0], dtype=torch.float32)
        if img_class == "FRESH":
            label = torch.tensor(2, dtype=torch.long)
        elif img_class == "HALF":
            label = torch.tensor(1, dtype=torch.long)
        elif img_class == "SPOILED":
            label = torch.tensor(0, dtype=torch.long)
            
        return img, label
       
class MeatTestDataset(Dataset):
    """A PyTorch Dataset class for loading images from a directory containing meat freshness images.
    
    Args:
        data_dir (str): The path to the directory containing the image files.
        transform (callable, optional): Optional transforms to be applied to the images.
    """
    def __init__(self, data_dir, transform=None):
      """
        Initializes a new instance of the MeatDataset class.
        
        Args:
            data_dir (str): The path to the directory containing the image files.
            transform (callable, optional): Optional transforms to be applied to the images.
            """
      self.data_dir = data_dir
      self.file_names = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
      self.transform = transform

    def __len__(self):
      """
        Returns the number of images in the dataset.
        """
      return len(self.file_names)

    def __getitem__(self, idx):
       """
        Returns the image and corresponding label at the given index in the dataset.
        
        Args:
            idx (int): The index of the image to retrieve.
            
        Returns:
            tuple: A tuple containing the image, corresponding label, and file name.
        """
       file_name = self.file_names[idx]
       img_path = os.path.join(self.data_dir, file_name)
       img_class = file_name.split("-")[0]

       # Load the image
       img = Image.open(img_path)
       # Apply the transforms
       if self.transform:
        img = self.transform(img)

        # Convert the class label to a tensor
        if img_class == "FRESH":
            label = torch.tensor(2, dtype=torch.long)
        elif img_class == "HALF":
            label = torch.tensor(1, dtype=torch.long)
        elif img_class == "SPOILED":
            label = torch.tensor(0, dtype=torch.long)
            
        return img, label, file_name