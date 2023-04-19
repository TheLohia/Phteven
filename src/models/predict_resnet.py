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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# import sys
# sys.path.append("~/bt5153/src")
# from src.utils import get_constants, update_constants, delete_constants
# from src.features.build_features import MeatDataset

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
    
def predict_model(model, dataloaders, criterion):
    """Predict the labels for a given PyTorch model using the specified data loaders and loss function.

    Args:
        model (torch.nn.Module): PyTorch model to be used for prediction.
        dataloaders (dict): Dictionary containing one or more PyTorch data loaders for the "test" phase of prediction.
        criterion (torch.nn.modules.loss._Loss): PyTorch loss function to be used for calculating the prediction error.

    Returns:
        A pandas DataFrame containing the predicted labels and corresponding true labels for the test data. Logs 
        the final accuracy, precision, recall, F1 scores are logged to Weights & Biases using the wandb.log function.
    """


    since = time.time()

    val_acc_history = []
    best_acc = 0.0

    predicts = []

    # Each epoch has a training and validation phase
    for phase in ['test']:
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

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                # Get model outputs and calculate loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                for i in range(labels.size(0)):
                    predicts.append({
                        "predict": preds[i].item(),
                        "label": labels[i].item()
                    })

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            running_steps += 1
            running_samples += inputs.size(0)

            test_loss = running_loss / len(dataloaders[phase].dataset)
            test_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            epoch_test_metrics = {"test_loss": test_loss}
            wandb.log({**epoch_test_metrics})

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, test_loss, test_acc))

            # deep copy the model
            if phase == 'test' and test_acc > best_acc:
                best_acc = test_acc
            if phase == 'test':
                val_acc_history.append(test_acc)

        print()
    predict_df = pd.DataFrame(predicts)
    final_accuracy = accuracy_score(predict_df['label'], predict_df['predict'])
    precision, recall, f1_score, support = precision_recall_fscore_support(predict_df['label'], predict_df['predict'], average="weighted")
    test_metrics = {"test_accuracy": final_accuracy,
                    "test_precision": precision,
                    "test_recall": recall,
                    "test_f1_score": f1_score,
                    "suport": support
        
    }
    wandb.log({**test_metrics})
    time_elapsed = time.time() - since
    print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return predict_df


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
        """ Resnet
        """
        #model_ft = models.resnet18(weights=weights)
        model_ft = models.resnet50(weights=weights)
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

    # Define the transform for the validation data
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    with open("./nnc_wandb_key.txt", "r") as f: # change this to your own API key
        wandb_key = f.read()

    wandb.login(key=wandb_key)

    dir = sys.argv[1]

    # train_dir = f"{dir}/raw/train_split/train"
    # val_dir = f"{dir}/raw/train_split/val"
    test_dir = f"{dir}/data/raw/test"

    # Will add to constants.json later the the hyperparameteres used here
    model_name = "resnet"
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    #WEIGHTS = "ResNet18_Weights.IMAGENET1K_V1" #ResNet18 weights
    WEIGHTS = "ResNet50_Weights.IMAGENET1K_V2" #ResNet50 weights
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True
    #feature_extract = False

    criterion = nn.CrossEntropyLoss()

    meat_datasets = {
                "test": MeatDataset(test_dir, transform=test_transform)
                }

    meat_loaders ={
                "test": DataLoader(meat_datasets["test"], 
                                    batch_size=BATCH_SIZE, shuffle=False)}
    
    
    # Initialize the model for this run
    trained_model, input_size = initialize_model(model_name, NUM_CLASSES, feature_extract, weights=WEIGHTS)
    model_path = "/home/c/casanath/bt5153/models/resnet50.pth" #change file name here
    trained_model.load_state_dict(torch.load(model_path, map_location=device))
    trained_model.to(device)

    wandb.init(
    project="bt5153_resnet",
    group = "test_results",
    mode="online",
        config={
            "batch_size": BATCH_SIZE,
            "loss": criterion,
            })
    
    predict_df = predict_model(trained_model, meat_loaders, criterion)
    wandb.finish()

    #predict_df.to_csv("/home/c/casanath/bt5153/models/resnet18_test_predict.csv")
    predict_df.to_csv("/home/c/casanath/bt5153/models/resnet50_test_predict.csv")