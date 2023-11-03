import torch
import torchvision.models as models

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define a sample input image (you should replace this with your image data)
input_image = torch.randn(1, 3, 600, 600)  # Batch size of 1, 3 channels (RGB), 224x224 pixels


res = model(input_image)  # Forward pass through the network
print(f'res is of shape: {res.shape}')

# Pass the input image through the model
with torch.no_grad():
    x = input_image
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.modules.linear.Linear):
            x = torch.flatten(x, 1)
            print(f"Layer: Flatten, Feature Map Size: {x.size()}")
        x = layer(x)
        print(f"Layer: {name}, Feature Map Size: {x.size()}")



