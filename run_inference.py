import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

from resnet import ResNet
from data_loader import get_data_loaders, plot_images
from utils import calculate_normalisation_params
from train import evaluate

from PIL import ImageDraw
from PIL import ImageFont
import cv2


import warnings
warnings.filterwarnings('ignore')

data_dir = 'data/cifar10'
batch_size = 128

# Normalisation parameters fo CIFAR10
means = [0.4918687901200927, 0.49185976472299225, 0.4918583862227116]
stds  = [0.24697121702736, 0.24696766978537033, 0.2469719877121087]

normalize = transforms.Normalize(
    mean=means,
    std=stds,
)

train_transform = transforms.Compose([
    # 4 pixels are padded on each side, 
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    # For testing, we only evaluate the single 
    # view of the original 32×32 image.
    transforms.ToTensor(),
    normalize
])

_, test_loader = get_data_loaders(data_dir,
                                  batch_size,
                                  train_transform,
                                  test_transform,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)

n=7
model = ResNet(n, shortcuts=True)
results_file = f'results/resnet{6*n+2}.csv' # these files were produced Dr. Lim when I ran the training.
model_file = f'pretrained/resnet{6*n+2}.pt'
model.load_state_dict(torch.load(model_file))
model.to('cuda')

# open file opener for image files
from tkinter import Tk
from tkinter.filedialog import askopenfilename
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
# inference on a single image
from PIL import Image
from torchvision import transforms
orig_image = Image.open(filename)
img = test_transform(orig_image)
img = img.unsqueeze(0)
img = img.to('cuda')
model.eval()
output = model(img)
_, pred = torch.max(output, 1)
categories = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
obj_class = categories[pred.item()]
draw = ImageDraw.Draw(orig_image)
font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
font = ImageFont.truetype(font_path, size=32)
draw.text((0, 0), obj_class, (255, 0, 0), font=font)
img = test_transform(orig_image)
img = img.unsqueeze(0)
img = img.to('cuda')

# show image and prediction
plt.imshow(img.squeeze().cpu().numpy().transpose(1,2,0))
plt.title(obj_class)
plt.show()

