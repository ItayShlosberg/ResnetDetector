import json
from Models.modified_resnet import *
import os
import numpy as np
import matplotlib.pyplot as plt
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
import torchvision.transforms.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR


class AffineTransform:
    def __init__(self, angle, translate=(0, 0), scale=1, shear=0):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, img):
        return F.affine(img, self.angle, self.translate, self.scale, self.shear)

class PatchesDataset(Dataset):
    def __init__(self, root_dir, training_mode, patch_size):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        # Define a transform for data augmentation
        augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Flip the image horizontally with a probability of 0.5
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Apply color jitter
            transforms.RandomCrop(patch_size), # randomresizedcrop , scale=(1.0, 1.0), ratio=(1.0, 1.0)),  # Randomly crop a 64x64 patch
            transforms.RandomRotation(20)
            # AffineTransform(angle=0, translate=(3, 3), scale=1, shear=0),
            # transforms.CenterCrop(patch_size)
            ])

        # Define a transform to normalize the data
        validate_transform = transforms.Compose([
            # transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform = transforms.Compose([augmentation_transform, validate_transform]) if training_mode else validate_transform

        self.negative_files = [os.path.join(root_dir, 'negative', f) for f in
                               os.listdir(os.path.join(root_dir, 'negative'))]
        self.positive_files = [os.path.join(root_dir, 'positive', f) for f in
                               os.listdir(os.path.join(root_dir, 'positive'))]
        self.all_files = self.negative_files + self.positive_files
        self.labels = [0] * len(self.negative_files) + [1] * len(self.positive_files)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_name = self.all_files[idx]
        image = Image.open(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# First version
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
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
#             image = self.transform(image)
#
#         return image, img_name


class PadImage(object):
    def __init__(self, desired_size):
        self.desired_size = desired_size

    def __call__(self, img):
        delta_width = self.desired_size[1] - img.width
        delta_height = self.desired_size[0] - img.height
        padding = (
        delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
        return transforms.functional.pad(img, padding)


class RawDataset(Dataset):
    # NB resnet_torch_to_js version
    def __init__(self, root_dir, annotation_file=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir


        # Define a transform to normalize the data
        validate_transform = transforms.Compose([
            # transforms.Resize((64, 64)),
            PadImage((640, 640)), # Highly problematic
            transforms.ToTensor(),
             # PadToSize((640, 640)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: x.unsqueeze(0))

        ])

        self.transform = validate_transform

        self.all_files =[os.path.join(root_dir, f) for f in
                               os.listdir(root_dir)]

        self.annotations = None
        if annotation_file:
            self.annotations = self.load_annotations(annotation_file, only_person=True)

    def load_annotations(self, annotation_file, only_person=True):
        # Load the JSON file
        with open(annotation_file, 'r') as json_file:
            annotations = json.load(json_file)

        classes = {val['id']: val['name'] for val in annotations['categories']}

        image_dict = {image['id']: {**image, **{'annotations': []}} for image in annotations['images']}
        for ann in annotations['annotations']:
            if (only_person and classes[ann['category_id']] == 'person') or not only_person:
                image_dict[ann['image_id']]['annotations'] += [
                    (ann['bbox'], ann['category_id'], classes[ann['category_id']])]

        annotations = {image_data['file_name']: image_data for image_id, image_data in
                      image_dict.items()}  # only images with person instances
        return annotations


    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_name = self.all_files[idx]
        image = Image.open(img_name)
        annotations = None

        if self.transform:
            image_tensor = None # self.transform(image) # TODO: For now, the tensor does not represent what it should have - the model return an invalid output on this tensor

        if not self.annotations is None:
            annotations = self.annotations[img_name.split('\\')[-1]]
            bboxes = np.array([bbox[0] for bbox in annotations['annotations']])

        return image, image_tensor, img_name, bboxes
