import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
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


def draw_ann_on_image(image, annotations):
    # bbox of xywh -> xy: top-left

    # Create a figure and an Axes object
    fig, ax = plt.subplots()


    for bbox in annotations:
        x, y, width, height = bbox
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.imshow(image)


def draw_grid_on_image(image, n_rows, n_cols):
    width, height = image.size
    # print(image.size)
    # Calculate the width and height of each grid cell
    cell_width = width // n_cols
    cell_height = height // n_rows

    fig, ax = plt.subplots()
    # Draw red grid lines on the image
    for i in range(1, n_cols):
        x = i * cell_width
        ax.axvline(x, color='red', linewidth=2)
    for i in range(1, n_rows):
        y = i * cell_height
        ax.axhline(y, color='red', linewidth=2)

    ax.imshow(image)
    plt.show()


def draw_ann_and_grid_on_image(image, annotations, n_rows, n_cols, output_name=None):
    width, height = image.size
    # Calculate the width and height of each grid cell
    cell_width = width // n_cols
    cell_height = height // n_rows


    # Create a figure and an Axes object
    fig, ax = plt.subplots()

    plt.title('Image');

    for detection in annotations:
        # Assume each detection is a dictionary with 'bbox' key
        x, y, width, height = detection
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    for i in range(1, n_cols):
        x = i * cell_width
        ax.axvline(x, color='red', linewidth=1)
    for i in range(1, n_rows):
        y = i * cell_height
        ax.axhline(y, color='red', linewidth=1)

    plt.imshow(image)
    if output_name:
        plt.savefig(output_name)


def display_predictions(image, probability_matrix):
    # Convert image to numpy array
    img_array = np.asarray(image).copy()

    # Get image dimensions
    height, width, _ = img_array.shape

    # Get grid size
    N = probability_matrix.shape[0]

    # Calculate the size of each cell
    cell_height = height // N
    cell_width = width // N

    # Apply the probability matrix to the image
    for i in range(N):
        for j in range(N):
            # Get the current cell
            cell = img_array[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]

            # Apply the probability factor
            img_array[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width] = cell * \
                                                                                                    probability_matrix[
                                                                                                        i, j]

    # Convert back to PIL Image
    transformed_image = Image.fromarray(img_array.astype('uint8'))

    # Create a subplot with 2 axes side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Display the transformed image
    axes[1].imshow(transformed_image)
    axes[1].set_title('Transformed Image')
    axes[1].axis('off')


def display_predictions_and_ann(image, probability_matrix, n_rows, n_cols, annotations):
    # Convert image to numpy array
    img_array = np.asarray(image).copy()

    # Get image dimensions
    height, width, _ = img_array.shape

    # Get grid size
    N = probability_matrix.shape[0]

    # Calculate the size of each cell
    cell_height = height // N
    cell_width = width // N

    # Apply the probability matrix to the image
    for i in range(N):
        for j in range(N):
            # Get the current cell
            cell = img_array[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]

            # Apply the probability factor
            img_array[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width] = cell * \
                                                                                                    probability_matrix[
                                                                                                        i, j]

    # Convert back to PIL Image
    transformed_image = Image.fromarray(img_array.astype('uint8'))

    # Create a subplot with 2 axes side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display the original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Display the transformed image
    axes[1].imshow(transformed_image)
    axes[1].set_title('Transformed Image')
    axes[1].axis('off')

    axes[2].set_title('Annotations')
    ####################################
    ax = axes[2]
    width, height = image.size
    # Calculate the width and height of each grid cell
    cell_width = width // n_cols
    cell_height = height // n_rows

    ax.imshow(image)

    for detection in annotations:
        # Assume each detection is a dictionary with 'bbox' key
        x, y, width, height = detection
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    for i in range(1, n_cols):
        x = i * cell_width
        ax.axvline(x, color='red', linewidth=1)
    for i in range(1, n_rows):
        y = i * cell_height
        ax.axhline(y, color='red', linewidth=1)

    # if output_name:
    #     plt.savefig(output_name)


def adjust_bbox_to_padding(image, img_bboxes, padded_image_size):
    if len(img_bboxes) == 0:
        return img_bboxes
    width, height = image.size
    img_bboxes[:, 0] += (padded_image_size - width)/2
    img_bboxes[:, 1] += (padded_image_size - height)/2
    return img_bboxes