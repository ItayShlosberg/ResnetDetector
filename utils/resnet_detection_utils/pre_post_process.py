import sys
from Models.modified_resnet import *
from Models.modified_resnet import *
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from torch.utils.data import random_split
import os
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import time



def make_patches(tensor):
    # (1, 3, 640, 640) -> (25, 3, 128, 128)

    # Use unfold to extract patches
    patches = tensor.unfold(2, 128, 128).unfold(3, 128, 128)

    # Permute and reshape to get the desired tensor shape
    patches = patches.permute(2, 3, 0, 1, 4, 5).reshape(25, 3, 128, 128)

    return patches

