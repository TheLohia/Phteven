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
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from flask import Flask, render_template, Response, request, send_file
import cv2

app = Flask(__name__)

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

def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x

def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x

def process_img(img):
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        torchvision.transforms.Lambda(nhwc_to_nchw),
        #torchvision.transforms.Lambda(lambda x: x*(1/255)),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        torchvision.transforms.Lambda(nchw_to_nhwc)
            ])
    return test_transform(img)

def predict(img):
    img = torch.Tensor(img)
    img = nhwc_to_nchw(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    output = trained_model(img)
    output = torch.argmax(output)
    output = output.cpu().numpy()
    return output

@app.route('/', methods=['GET','POST'])
def home():
   return render_template('index.html')

@app.route('/formSubmit', methods=['POST'])
def submit():
	if(request.method == 'POST'):
            f = request.files['file']
            processed_img = process_img(Image.open(f))
            try:
                class_type = predict(processed_img)
                if class_type == 0:
                    output = "SPOILED"
                if class_type == 1:
                    output = "HALF FRESH"
                if class_type == 2:
                    output = "FRESH"
            except Exception as e:
                error_msg='Unable to process image'
                return render_template('error_page.html', msg=error_msg)
            return render_template('result.html', class_type=output)
        
# Error handlers
@app.errorhandler(404)
def error404(e):
    return render_template("404.html")

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "resnet"
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    WEIGHTS = "ResNet50_Weights.IMAGENET1K_V2" #ResNet50 weights
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    trained_model, input_size = initialize_model(model_name, NUM_CLASSES, feature_extract, weights=WEIGHTS)
    model_path = "../models/resnet50_fe.pth" #change file name here
    trained_model.load_state_dict(torch.load(model_path, map_location=device))
    trained_model.to(device)
    
    app.run(threaded=True, port=8080)