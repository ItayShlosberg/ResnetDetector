##########################################
# Was used when we explored YOLO V8 NOT with resnet detection.

##########################################


import torchvision.transforms as transforms
import torch
from torchvision import transforms


class AddDimension(object):
    def __call__(self, tensor):
        return tensor.unsqueeze(0)  # Adds a dimension at the 0th position

# Define the padding transform
class PadToSize(object):
    def __init__(self, desired_size):
        self.desired_size = desired_size

    def __call__(self, img):
        delta_width = self.desired_size[1] - img.width
        delta_height = self.desired_size[0] - img.height
        padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
        return transforms.functional.pad(img, padding)


def create_onnx_preprocess_transform():
    # Define the transform composition
    transform = transforms.Compose([
        #     transforms.Resize((640, 640)),
        PadToSize((640, 640)),  # Pad image to 640x640
        transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        AddDimension()  # This will add an extra dimension
    ])
    return transform



