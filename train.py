import os
import time
import json
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# from workspace_utils import active_session
# import model_structure
from torch import nn, optim
from PIL import Image
from torchvision import datasets, transforms, models
from collections import OrderedDict

arg_parse = argparse.ArgumentParser(description='train.py')
arg_parse.add_argument('data_dir', action="store", default="./flowers")
arg_parse.add_argument('--gpu', dest="gpu", action="store", default="gpu")
arg_parse.add_argument('--save_dir', dest="save_dir", action="store", default="checkPoint.pth")
arg_parse.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
arg_parse.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
arg_parse.add_argument('--structure', dest="structure", action="store", default="vgg11", type = str)
arg_parse.add_argument('--hidden_layer', dest="hidden_layer", action="store", type=int, default=1024)


parse = arg_parse.parse_args()
data_dir = parse.data_dir
power = parse.gpu
checkpoint_path = parse.save_dir
lr = parse.learning_rate
epochs = parse.epochs
structure = parse.structure
hidden_layer = parse.hidden_layer
dataloaders = model_structure.load_data(data_dir)
structure, hidden_layer = model_structure.my_model(model, classifier, criterion, optimizer)
model_structure.train_model(model, criterion, optimizer, lr, epochs, power)
model.structure.save_checkpoint(checkpoint_path, structure)
