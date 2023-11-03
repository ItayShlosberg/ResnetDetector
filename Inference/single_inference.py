import sys
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
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader

class PadToSize(object):
    def __init__(self, desired_size):
        self.desired_w, self.desired_h = desired_size

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        padding = transforms.Pad((0, 0, int((self.desired_w-w)), int((self.desired_h-h))), padding_mode='constant')
        padded_img = padding(img)
        return padded_img

# model
class RawDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir


        # Define a transform to normalize the data
        validate_transform = transforms.Compose([
            # transforms.Resize((64, 64)),
            transforms.ToTensor(),
            PadToSize((640, 640)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: x.unsqueeze(0))

        ])

        self.transform = validate_transform

        self.all_files =[os.path.join(root_dir, f) for f in
                               os.listdir(root_dir)]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_name = self.all_files[idx]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, img_name


if __name__ == '__main__':
    OUTPUT_FOLDER = r'C:\Users\itay\Desktop\IDF\models\train\1.11.23'

    model = BinaryModifiedResNet18(num_classes=2, grid_size=5)

    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load(join(OUTPUT_FOLDER, 'checkpoint.pt')))

    raw_dataset = RawDataset(fr'C:\Users\itay\Desktop\IDF\datasets\patches_coco_val\raw_data\Images')

    tensor_input, image_name = raw_dataset[0]
    print(tensor_input.shape)
    model(tensor_input, inference=True)
    # model(torch.unsqueeze(tensor_input, dim=0), Inference=True)