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

MODEL_FOLDER = r'C:\Users\itay\Desktop\IDF\models\train\1.11.23'
OUTPUT_FOLDER = r'C:\Users\itay\Desktop\IDF\models\train\1.11.23'
IMAGE_NUM = 5

# def make_patches(tensor):
#     # (1, 3, 640, 640) -> (25, 3, 128, 128)
#
#     # Use unfold to extract patches
#     patches = tensor.unfold(2, 128, 128).unfold(3, 128, 128)
#
#     # Permute and reshape to get the desired tensor shape
#     patches = patches.permute(2, 3, 0, 1, 4, 5).reshape(25, 3, 128, 128)
#
#     return patches
#
#
#
#
# class PadImage(object):
#     def __init__(self, desired_size):
#         self.desired_size = desired_size
#
#     def __call__(self, img):
#         delta_width = self.desired_size[1] - img.width
#         delta_height = self.desired_size[0] - img.height
#         padding = (
#         delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
#         return transforms.functional.pad(img, padding)
#
# class RawDataset(Dataset):
#     def __init__(self, root_dir):
#         """
#         Args:
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.root_dir = root_dir
#
#
#         # Define a transform to normalize the data
#         validate_transform = transforms.Compose([
#             # transforms.Resize((64, 64)),
#             PadImage((640, 640)),
#             transforms.ToTensor(),
# #             PadToSize((640, 640)),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             transforms.Lambda(lambda x: x.unsqueeze(0))
#
#         ])
#
#         self.transform = validate_transform
#
#         self.all_files =[os.path.join(root_dir, f) for f in
#                                os.listdir(root_dir)]
#
#     def __len__(self):
#         return len(self.all_files)
#
#     def __getitem__(self, idx):
#         img_name = self.all_files[idx]
#         image = Image.open(img_name)
#
#         if self.transform:
#             image_tensor = self.transform(image)
#
#         return image, image_tensor, img_name


if __name__ == '__main__':

    model = BinaryModifiedResNet18(num_classes=2, grid_size=5)

    # Load the last checkpoint with the best model
    # model.to('cuda')
    model.load_state_dict(torch.load(join(MODEL_FOLDER , 'checkpoint_ep100.pt')))

    raw_dataset = RawDataset(fr'C:\Users\itay\Desktop\IDF\datasets\COCO\val2017')



    image, image_tensor, img_name = raw_dataset[IMAGE_NUM] # TODO: Image tensor is not in the right format and therefore we will use the image itself

    transform = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    start_time = time.perf_counter()
    image_tensor = transform(np.array(PadImage((640, 640))(image))).unsqueeze(0) # That is the right way to transform the tensor
    image_tensor = make_patches(image_tensor) # converet: (1, 3, 640, 640) -> (25, 3, 128, 128)
    end_time = time.perf_counter()
    print(f"transform time in milliseconds: {(end_time - start_time) * 1000:.2f}")

    start_time = time.perf_counter()
    model.eval()
    model_output = model(image_tensor) # (25, 1) -->
    end_time = time.perf_counter()
    print(f"model time in milliseconds: {(end_time - start_time) * 1000:.2f}")


    model.to('cuda')
    image_tensor  =image_tensor.to('cuda')
    model.eval()
    start_time = time.perf_counter()
    model_output = model(image_tensor) # (25, 1) -->
    end_time = time.perf_counter()
    print(f"model time in milliseconds: {(end_time - start_time) * 1000:.2f}")

    start_time = time.perf_counter()
    model_output = model(image_tensor) # (25, 1) -->
    end_time = time.perf_counter()
    print(f"model time in milliseconds: {(end_time - start_time) * 1000:.2f}")

    start_time = time.perf_counter()
    model_output = model(image_tensor) # (25, 1) -->
    end_time = time.perf_counter()
    print(f"model time in milliseconds: {(end_time - start_time) * 1000:.2f}")

    start_time = time.perf_counter()
    model_output = model(image_tensor) # (25, 1) -->
    end_time = time.perf_counter()
    print(f"model time in milliseconds: {(end_time - start_time) * 1000:.2f}")

    # process_image(PadImage((640, 640))(image), model_output.reshape(5,5).to('cpu').detach().numpy())

    # plt.savefig(join(OUTPUT_FOLDER, r'model_prediction.jpg'))
