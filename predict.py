import os
import time
import json
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from workspace_utils import active_session
import model_structure
from torch import nn, optim
from PIL import Image
from torchvision import datasets, transforms, models
from collections import OrderedDict


arg_parse = argparse.ArgumentParser(description='predict.py')
arg_parse.add_argument('img', nargs='*', action="store", type=str, default="flowers/test/28/image_05214.jpg")
arg_parse.add_argument('--gpu', dest="gpu", action="store", default="gpu")
arg_parse.add_argument('checkpoint', nargs='*', action="store", type=str, default="checkPoint.pth")
arg_parse.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
arg_parse.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')

parse = arg_parse.parse_args()
number_of_outputs = parse.top_k
power = parse.gpu
input_img =parse.img
checkpoint_path = parse.checkpoint

dataloaders = model_structure.load_data()
model_structure.load_checkpoint(checkpoint_path)


if parse.GPU:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

with open(category_names) as json_file:
    cat_to_name = json.load(json_file)

top_prob, top_classes = model_structure.predict(input_image, model, number_of_outputs)

for i in range(len(top_prob)):
    print(json_file"{cat_to_name[top_classes[i]]:<25} {top_prob[i]*100:.3f}%")
