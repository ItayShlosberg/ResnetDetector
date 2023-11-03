import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import os
from os.path import join
import numpy as np



IMAGE_FOLDER = fr'C:\Users\itay\Desktop\IDF\datasets\COCO\val2017'
OUTPUT_FOLDER = fr'C:\Users\itay\Desktop\IDF\datasets\patches_coco_val\EXP\processed_data_all'
POSITIVE_FOLDER_NAME = fr'positive'
NEGATIVE_FOLDER_NAME = fr'negative'
ANNOTATION_PATH = fr'C:\Users\itay\Desktop\IDF\datasets\COCO\annotations_trainval2017\annotations\instances_val2017.json'

# Define the number of rows and columns for the grid
N_ROWS = 5
N_COLS = 5
DESIRED_DIMS = (640, 640)           # The dims of the Inference images

# The content fraction needed for a cell to be considered positive
MIN_POSITIVE_OBJECT_SCORE = 0.2     # 20% of the cell - overlap between any annotation that surpasses will be considered as positive
max_negative_object_score = 0.05    # 5% of the cell - lack of overlap between any annotation will be considered as negative
BLACK_PATCHES_THRESHOLD = 0.1       # 10% - a patch whose more than 10% of its content is black will not be saved
PADDING_ADDITION = 3                # How many pixel to add with padding for augmentations - width += PADDING_ADDITION * 2, height += PADDING_ADDITION * 2


def add_padding(original_image, desired_size):
    # Create a new image with the desired size and white background
    new_image = Image.new("RGB", desired_size, (0, 0, 0))

    # Calculate the position to paste the original image with padding
    x_offset = (desired_size[0] - original_image.width) // 2
    y_offset = (desired_size[1] - original_image.height) // 2

    # Paste the original image onto the new image with padding
    new_image.paste(original_image, (x_offset, y_offset))

    return new_image #.show()  # Display the padded image


def load_annotations(annotation_path):
    # Load the JSON file
    with open(annotation_path, 'r') as json_file:
        annotations = json.load(json_file)

    classes = {val['id']: val['name'] for val in annotations['categories']}

    image_dict = {image['id']: {**image, **{'annotations': []}} for image in annotations['images']}
    for ann in annotations['annotations']:
        if classes[ann['category_id']] == 'person':
            image_dict[ann['image_id']]['annotations'] += [
                (ann['bbox'], ann['category_id'], classes[ann['category_id']])]

    image_dict = {image_data['file_name']: image_data for image_id, image_data in
                  image_dict.items()}  # only images with person instances
    person_images = {image_name: image_metadata for image_name, image_metadata in image_dict.items() if
                     len(image_metadata['annotations']) > 0}

    return person_images, image_dict


def calculate_intersection_with_box1(box1, box2):
    # Extract coordinates from the input boxes
    b1_x1, b1_x2, b1_y1, b1_y2 = box1
    b2_x1, b2_x2, b2_y1, b2_y2 = box2

    # Calculate the intersection coordinates
    intersection_x1 = max(b1_x1, b2_x1)
    intersection_x2 = min(b1_x2, b2_x2)
    intersection_y1 = max(b1_y1, b2_y1)
    intersection_y2 = min(b1_y2, b2_y2)

    # Calculate the area of intersection
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)

    # Calculate the areas of each box
    box1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    box2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    # Calculate the IoU
    intersection = intersection_area / box1_area #float(box1_area + box2_area - intersection_area)

    return intersection


def save_patches_images(cropped_positive_images, cropped_negative_images):
    if not os.path.isdir(os.path.join(OUTPUT_FOLDER, POSITIVE_FOLDER_NAME)):
        os.makedirs(os.path.join(OUTPUT_FOLDER, POSITIVE_FOLDER_NAME))

    if not os.path.isdir(os.path.join(OUTPUT_FOLDER, NEGATIVE_FOLDER_NAME)):
        os.makedirs(os.path.join(OUTPUT_FOLDER, NEGATIVE_FOLDER_NAME))

    for i, patch_image in enumerate(cropped_positive_images):
        output_path = os.path.join(OUTPUT_FOLDER, POSITIVE_FOLDER_NAME, f'{image_name}_{i + 1}.jpg')
        patch_image.save(output_path)

    for i, patch_image in enumerate(cropped_negative_images):
        if np.sum(np.array(patch_image) == 0) / len(
                np.array(patch_image).reshape(-1)) > BLACK_PATCHES_THRESHOLD:  # Black patched (from the padding)
            continue
        output_path = os.path.join(OUTPUT_FOLDER, NEGATIVE_FOLDER_NAME,
                                   f'{image_name}_{i + 1 + len(cropped_positive_images)}.jpg')
        patch_image.save(output_path)


def create_patches_from_images(object_probs_table, cell_location_table):
    # STEP 2:  create cropped_positive_images and  cropped_negative_images
    cropped_positive_images = []
    cropped_negative_images = []
    for row_idx in range(N_ROWS):
        for col_idx in range(N_COLS):
            # If object score is more than min_positive_object_score, which means that there is a big enough overlap with an object, so we will add it to the positive patches
            if object_probs_table[row_idx, col_idx] > MIN_POSITIVE_OBJECT_SCORE:
                (x1, y1, x2, y2) = cell_location_table[row_idx][col_idx]
                cropped_image = image.crop(
                    (x1 - PADDING_ADDITION, y1 - PADDING_ADDITION, x2 + PADDING_ADDITION, y2 + PADDING_ADDITION))
                cropped_positive_images.append(cropped_image)

            # If object score is less than max_negative_object_score, which means that there is not even a small overlap with an object, so we will add it to the negative patches
            if object_probs_table[row_idx, col_idx] < max_negative_object_score:
                (x1, y1, x2, y2) = cell_location_table[row_idx][col_idx]
                cropped_image = image.crop(
                    (x1 - PADDING_ADDITION, y1 - PADDING_ADDITION, x2 + PADDING_ADDITION, y2 + PADDING_ADDITION))
                cropped_negative_images.append(cropped_image)

    return cropped_positive_images, cropped_negative_images


def create_object_scores_table_and_cell_location_table():
    object_scores_table = np.zeros((N_ROWS, N_COLS))
    cell_location_table = np.zeros((N_ROWS, N_COLS)).tolist()

    for row_idx in range(N_ROWS):
        y1 = row_idx * cell_height
        y2 = (row_idx + 1) * cell_height
        for col_idx in range(N_COLS):
            x1 = col_idx * cell_width
            x2 = (col_idx + 1) * cell_width
            cell_location_table[row_idx][col_idx] = (x1, y1, x2, y2)
            #         print((x1,y1,x2,y2))

            max_obj_detection_intersection_score = 0
            for object_location in annotation_list:
                obj_x1, obj_y1, obj_width, obj_height = object_location
                obj_x2, obj_y2 = obj_x1 + obj_width, obj_y1 + obj_height

                intersection_in_cell = calculate_intersection_with_box1((x1, x2, y1, y2),
                                                                        (obj_x1, obj_x2, obj_y1, obj_y2))
                max_obj_detection_intersection_score = max(max_obj_detection_intersection_score,
                                                           intersection_in_cell)

            #             print(intersection_in_cell)
            object_scores_table[row_idx, col_idx] = max_obj_detection_intersection_score

    return object_scores_table, cell_location_table



if __name__ == '__main__':

    person_images, image_dict = load_annotations(ANNOTATION_PATH)
    images_names = os.listdir(IMAGE_FOLDER)
    for image_name in person_images.keys():
        print(image_name)
        image = Image.open(join(IMAGE_FOLDER, image_name))
        image_ann = image_dict[image_name]
        annotation_list = np.array(
            [ann[0] for ann in image_ann['annotations']])  # x_top_left, y_top_left, width, height
        if image.size[0] > DESIRED_DIMS[0] or image.size[1] > DESIRED_DIMS[1]:
            print(f"desired_dims are not good!!!!! {image_name}, {image.size}")

        # Padding
        label_offset_w = (DESIRED_DIMS[0] - image.size[0]) / 2
        label_offset_h = (DESIRED_DIMS[1] - image.size[1]) / 2
        annotation_list[:, 0] += label_offset_w
        annotation_list[:, 1] += label_offset_h
        image = add_padding(image, DESIRED_DIMS)

        width, height = image.size
        print(image.size)
        # Calculate the width and height of each grid cell
        cell_width = width // N_COLS
        cell_height = height // N_ROWS

        # STEP 1: Create object_probs_table and cell_location_table
        object_scores_table, cell_location_table = create_object_scores_table_and_cell_location_table()

        # STEP 2:  create cropped_positive_images and  cropped_negative_images
        cropped_positive_images, cropped_negative_images = create_patches_from_images(object_scores_table,
                                                                                      cell_location_table)


        # STEP 3:  Save each image to the output folder
        save_patches_images(cropped_positive_images, cropped_negative_images)