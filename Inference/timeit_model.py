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




def process_image(image, probability_matrix):
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
