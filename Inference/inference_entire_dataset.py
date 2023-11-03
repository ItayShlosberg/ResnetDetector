from Data.Datasets import *
from utils.resnet_detection_utils.pre_post_process import *
from utils.visualizations import *

MODEL_FOLDER = r'C:\Users\itay\Desktop\IDF\models\train\3.11.23'
OUTPUT_FOLDER = r'C:\Users\itay\Desktop\IDF\models\train\3.11.23\inference'
MODEL_FILE_NAME = r'checkpoint_ep200.pt'
IMAGES_DIR = fr'C:\Users\itay\Desktop\IDF\datasets\COCO\val2017'
ANNOTATION_FILE = fr'C:\Users\itay\Desktop\IDF\datasets\COCO\annotations_trainval2017\annotations\instances_val2017.json'
DEVICE = 'cuda'
MODEL_IMAGE_SIZE = 640
GRID_SIZE = 5
ADD_PADDING = True
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
# def process_image(image, probability_matrix):
#     # Convert image to numpy array
#     img_array = np.asarray(image).copy()
#
#     # Get image dimensions
#     height, width, _ = img_array.shape
#
#     # Get grid size
#     N = probability_matrix.shape[0]
#
#     # Calculate the size of each cell
#     cell_height = height // N
#     cell_width = width // N
#
#     # Apply the probability matrix to the image
#     for i in range(N):
#         for j in range(N):
#             # Get the current cell
#             cell = img_array[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
#
#             # Apply the probability factor
#             img_array[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width] = cell * \
#                                                                                                     probability_matrix[
#                                                                                                         i, j]
#
#     # Convert back to PIL Image
#     transformed_image = Image.fromarray(img_array.astype('uint8'))
#
#     # Create a subplot with 2 axes side by side
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#
#     # Display the original image
#     axes[0].imshow(image)
#     axes[0].set_title('Original Image')
#     axes[0].axis('off')
#
#     # Display the transformed image
#     axes[1].imshow(transformed_image)
#     axes[1].set_title('Transformed Image')
#     axes[1].axis('off')
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
    model.to(DEVICE)
    model.eval()

    # Load the last checkpoint with the best model
    # model.to('cuda')
    model.load_state_dict(torch.load(join(MODEL_FOLDER , MODEL_FILE_NAME)))

    raw_dataset = RawDataset(IMAGES_DIR, ANNOTATION_FILE)


    for image_idx in range(len(raw_dataset)):
    # for image_idx in range(1):  # len(raw_dataset)):
        print(f'Image num: {image_idx + 1}, out of {len(raw_dataset)}')
        image, image_tensor, img_name, img_bboxes = raw_dataset[image_idx] # TODO: Image tensor is not in the right format and therefore we will use the image itself
        if len(np.array(image).shape) != 3:
            continue

        transform = transforms.Compose([
            # transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image_tensor = transform(np.array(PadImage((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))(image))).unsqueeze(0).to(DEVICE) # That is the right way to transform the tensor
        image_tensor = make_patches(image_tensor)   # convert: (1, 3, 640, 640) -> (25, 3, 128, 128) (vector of patches

        model_output = model(image_tensor)
        model_output = model_output.reshape(5, 5).to('cpu').detach().numpy()

        if ADD_PADDING:
            img_bboxes = adjust_bbox_to_padding(image, img_bboxes, MODEL_IMAGE_SIZE )
            image = PadImage((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))(image)


        # display_predictions(image, model_output)
        # draw_ann_and_grid_on_image(image, img_bboxes, GRID_SIZE, GRID_SIZE, join(OUTPUT_FOLDER, fr'pred_{image_idx}.jpg'))
        display_predictions_and_ann(image, model_output, GRID_SIZE, GRID_SIZE, img_bboxes)

        plt.savefig(join(OUTPUT_FOLDER, fr'pred_{image_idx}.jpg'))