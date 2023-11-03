import os
import shutil
import math

source_directory = r'' # input("Enter the source directory path (containing images): ")
destination_directory = r'' # input("Enter the destination directory path (for organized subfolders): ")

def organize_images(source_dir, destination_dir, images_per_subfolder=2000):
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # List all the image files in the source directory
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    # Calculate the number of subfolders needed
    num_subfolders = math.ceil(len(image_files) / images_per_subfolder)

    # Create subfolders and move images
    for i in range(num_subfolders):
        subfolder_name = f"subfolder_{i + 1}"
        subfolder_path = os.path.join(destination_dir, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        # Calculate the start and end indices for the current subfolder
        start_idx = i * images_per_subfolder
        end_idx = min((i + 1) * images_per_subfolder, len(image_files))

        # Move the images to the current subfolder
        for j in range(start_idx, end_idx):
            src_file = os.path.join(source_dir, image_files[j])
            dst_file = os.path.join(subfolder_path, image_files[j])
            shutil.move(src_file, dst_file)
            print(f"Moved: {src_file} -> {dst_file}")

if __name__ == "__main__":

    organize_images(source_directory, destination_directory)
