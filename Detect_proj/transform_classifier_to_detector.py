import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image






def load_image():
    # Load an image
    image_path = fr'C:\Users\itays\Desktop\accumulated_files\images\im3.jpg'  # Replace with the path to your image
    image = Image.open(image_path)

    # Define the sliding window size and stride
    window_size = 224  # Size of the window (224x224 for ResNet-18)
    stride = 32  # Stride for sliding the window (adjust as needed)

    # Transform the image to match the model's input requirements
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


if __name__ == '__main__':


    # Load a pretrained ResNet-18 model
    resnet18 = models.resnet18(pretrained=True)

    # Remove the last classification layer
    model = nn.Sequential(*list(resnet18.children())[:-2])

    # Set the model to evaluation mode
    model.eval()


    image = load_image()
