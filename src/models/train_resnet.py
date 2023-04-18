import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import datasets, models, transforms
import time
import random
import torchvision
import copy
import wandb
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import sys
sys.path.append('../dev')
from src.evaluation import missclassification_cost
from src.utils import get_constants, update_constants, delete_constants
from src.features.build_features import MeatDataset
    
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    """Train the given PyTorch model using the specified data loaders, criterion, and optimizer.

    Args:
        model: PyTorch model to be trained.
        dataloaders: Dictionary containing one or more PyTorch data loaders for each of the "train"
            and "val" phases of training.
        criterion: PyTorch loss function to be optimized.
        optimizer: PyTorch optimizer to use for backpropagation.
        num_epochs: Number of epochs to train the model (default 25).

    Returns:
        A tuple containing the trained model and a list of validation accuracies per epoch.
    """

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_steps = 0
            running_samples = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc = phase):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_steps += 1
                running_samples += inputs.size(0)
                loss_step = running_loss/running_steps
                accuracy_step = running_corrects/running_samples
                if phase=="train":
                    train_metrics = {"train_loss": loss_step,
                                     "train_accuracy": accuracy_step}
                    wandb.log({**train_metrics})
                else:
                    val_metrics = {"val_loss": loss_step,
                                     "val_accuracy": accuracy_step}
                    wandb.log({**val_metrics})

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase=="train":
                epoch_train_metrics = {"epoch_train_loss": epoch_loss, 
                                       "epoch_train_accuracy": epoch_acc}
                wandb.log({**epoch_train_metrics})
            else:
                epoch_val_metrics = {"epoch_val_loss": epoch_loss, 
                                       "epoch_val_accuracy": epoch_acc}
                wandb.log({**epoch_val_metrics})

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    wandb.log({"best_accuracy": best_acc})

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def initialize_model(model_name, num_classes, feature_extract, weights):
    """
    Initializes a pre-trained model for fine-tuning on a new classification task.

    Args:
        model_name (str): Name of the model architecture to use, e.g., "resnet".
        num_classes (int): Number of classes in the new classification task.
        feature_extract (bool): Whether to freeze all model parameters except the final layer.
        weights (str): Path to the pre-trained weights file or one of {"imagenet", None}.

    Returns:
        model_ft (nn.Module): The initialized model with a new final layer for num_classes.
        input_size (int): The expected size of the input images for the model.
    """

    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size 

def set_parameter_requires_grad(model, feature_extracting):
    """
    Set requires_grad attribute of the parameters in the model to False if feature_extracting is True,
    allowing only the last layer to be learned during training if desired.

    Inputs:
        model (nn.Module): neural network model
        feature_extracting (bool): if True, only the last layer is learned during training

    Outputs:
        None
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

if __name__ == "__main__": 

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Define the transform for the validation data
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    with open("src/models/nnc_wandb_key.txt", "r") as f: # change this to your own API key
        wandb_key = f.read()

    wandb.login(key=wandb_key)

    train_dir = "data/raw/train_split/train"
    val_dir = "data/raw/train_split/val"

    # Will add to constants.json later the the hyperparameteres used here
    model_name = "resnet"
    
    # Number of classes in the dataset
    NUM_CLASSES = 3

    # Batch size for training (change depending on how much memory you have)
    BATCH_SIZE = 32

    # Number of epochs to train for
    EPOCHS = 2

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    #ResNet18 weights
    WEIGHTS ="ResNet18_Weights.IMAGENET1K_V1"

    LEARNING_RATE = 0.001

    criterion = nn.CrossEntropyLoss()

    meat_datasets = {"train": MeatDataset(train_dir, transform=train_transform),
                "val": MeatDataset(val_dir, transform=test_transform)}

    meat_loaders ={"train": DataLoader(meat_datasets["train"], 
                            batch_size=BATCH_SIZE, shuffle=True),
                "val": DataLoader(meat_datasets["val"], 
                                    batch_size=BATCH_SIZE, shuffle=False)}
    
    
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, NUM_CLASSES, feature_extract, weights=WEIGHTS)

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    
    optimizer_ft = optim.Adam(params_to_update, lr=LEARNING_RATE)

    wandb.init(
    project="bt5153_resnet",
    mode="disabled",
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "optimizer": "Adam",
            "loss": criterion,
            })
    
    model_ft, hist = train_model(model_ft, meat_loaders, criterion, optimizer_ft, num_epochs=EPOCHS)
    wandb.finish()

    # resnet_path = "models/resnet18.bin"
    # torch.save(model_ft, resnet_path)

    # class MeatDataset(Dataset):
#     """
#     Dataset class for the meat images.

#     Args:
#         data_dir (str): Path to the directory containing the image files.
#         transform (callable, optional): Optional transform to be applied on a sample.

#     Attributes:
#         file_names (list): List of file names for the image files.

#     """
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.file_names = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
#         self.transform = transform

#     def __len__(self):
#         """Returns the length of the dataset."""
#         return len(self.file_names)

#     def __getitem__(self, idx):
#         """
#         Returns a tuple of (image, label) for a given index.

#         Args:
#             idx (int): Index of the item to be returned.

#         Returns:
#             tuple: A tuple of (image, label).
#         """
#         file_name = self.file_names[idx]
#         img_path = os.path.join(self.data_dir, file_name)
#         img_class = file_name.split("-")[0]

#         # Load the image
#         img = Image.open(img_path)

#         # Apply the transforms
#         if self.transform:
#             img = self.transform(img)

#         # Convert the class label to a tensor
#         #label = torch.tensor([0, 0, 0], dtype=torch.long)
#         if img_class == "FRESH":
#             label = torch.tensor(2, dtype=torch.long)
#         elif img_class == "HALF":
#             label = torch.tensor(1, dtype=torch.long)
#         elif img_class == "SPOILED":
#             label = torch.tensor(0, dtype=torch.long)

#         return img, label