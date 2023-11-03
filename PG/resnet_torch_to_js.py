import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torchvision.models as models


class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedResNet18, self).__init__()
        # Load a pre-trained ResNet-18 model
        self.resnet18 = models.resnet18(pretrained=True)

        # Modify the fully connected layer
        self.resnet18.fc = nn.Identity()  # Remove the fully connected layer

        # Add a new fully connected layer for matrix multiplication
        self.fc = nn.Linear(512, num_classes, bias=True)

    def forward(self, x, inference=False):
        if not inference:
            # Training mode: Process patches as usual
            x = self.resnet18(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            # Inference mode: Process the entire image
            batch_size, C, H, W = x.size()
            assert H % 18 == 0 and W % 18 == 0, "Image dimensions should be divisible by 18"

            # Forward through all layers except the last fully connected layer
            x = self.resnet18.conv1(x)
            x = self.resnet18.bn1(x)
            x = self.resnet18.relu(x)
            x = self.resnet18.maxpool(x)
            x = self.resnet18.layer1(x)
            x = self.resnet18.layer2(x)
            x = self.resnet18.layer3(x)
            x = self.resnet18.layer4(x)
            x = self.resnet18.avgpool(x)

            # Reshape the feature map to associate each patch with one grid cell
            x = x.view(batch_size, 512, 18, 18)

            # Apply the fully connected layer as matrix multiplication
            x = x.view(batch_size, 512, -1)  # Flatten the spatial dimensions
            x = self.fc(x)

            # Optionally, apply softmax or any other activation function

        return x

class DEBUG__ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(DEBUG__ResNet, self).__init__()
        # Load a pre-trained ResNet-18 model
        self.resnet18 = models.resnet18(pretrained=True)

    # def forward(self, x, Inference=False):
    #     if not Inference:
    #         # Training mode: Process patches as usual
    #         x = self.resnet18(x)
    #         x = x.view(x.size(0), -1)
    #         x = self.fc(x)
    #     else:
    #         # Inference mode: Process the entire image
    #         batch_size, C, H, W = x.size()
    #         assert H % 18 == 0 and W % 18 == 0, "Image dimensions should be divisible by 18"
    #
    #         # Forward through all layers except the last fully connected layer
    #         x = self.resnet18.conv1(x)
    #         x = self.resnet18.bn1(x)
    #         x = self.resnet18.relu(x)
    #         x = self.resnet18.maxpool(x)
    #         x = self.resnet18.layer1(x)
    #         x = self.resnet18.layer2(x)
    #         x = self.resnet18.layer3(x)
    #         x = self.resnet18.layer4(x)
    #         x = self.resnet18.avgpool(x)
    #
    #         # Reshape the feature map to associate each patch with one grid cell
    #         x = x.view(batch_size, 512, 18, 18)
    #
    #         # Apply the fully connected layer as matrix multiplication
    #         x = x.view(batch_size, 512, -1)  # Flatten the spatial dimensions
    #         x = self.fc(x)
    #
    #         # Optionally, apply softmax or any other activation function
    #
    #     return x

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.resnet18.conv1(x)
        print(f'following conv1: {x.shape}')
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        print(f'following maxpool: {x.shape}')
        x = self.resnet18.layer1(x)
        print(f'following layer1: {x.shape}')
        x = self.resnet18.layer2(x)
        print(f'following layer2: {x.shape}')
        x = self.resnet18.layer3(x)
        print(f'following layer3: {x.shape}')
        x = self.resnet18.layer4(x)
        print(f'following layer4: {x.shape}')
        x = self.resnet18.avgpool(x)
        print(f'following avgpool: {x.shape}')
        x = torch.flatten(x, 1)
        print(f'following flatten: {x.shape}')
        x = self.resnet18.fc(x)
        print(f'following fc: {x.shape}')
        return x

modified_resnet = ModifiedResNet18(num_classes=2)
#
#
# dims = (1, 3, 648, 648)
# # dims = (1, 3, 36, 36)
# example_input = torch.randn(dims)
# modified_resnet(example_input, Inference=False)
# modified_resnet(example_input, Inference=True)



debug_resnet = DEBUG__ResNet(num_classes=2)
#
#
# dims = (1, 3, 648, 648)
# # dims = (1, 3, 36, 36)
# example_input = torch.randn(dims)

# in1 = torch.randn((1, 3, 640, 640))
# in2 = torch.randn((1, 3, 64, 64))

in1 = torch.randn((1, 3, 640, 640))
in2 = torch.randn((1, 3, 64, 64))

print(in1.shape)
res = debug_resnet(in1)
print()
print(in2.shape)
res = debug_resnet(in2)

# print(res)