import torch
import torch.nn as nn
import torchvision.models as models


# class NonbinaryModifiedResNet18(nn.Module):
#     def __init__(self, num_classes=2, grid_size=10, image_size=640):
#         super(ModifiedResNet18, self).__init__()
#         # Load a pre-trained ResNet-18 model
#         self.resnet18 = models.resnet18(pretrained=True)
#
#         self.grid_size = grid_size
#         patch_size = image_size/grid_size
#         layer_4_patch_size = int(patch_size * (1 / 2 ** 5))
#         # print(f"layer_4_patch_size: {layer_4_patch_size}")
#
#         # Modify the avgpool layer
#         self.resnet18.avgpool = nn.AvgPool2d(kernel_size=layer_4_patch_size, stride=layer_4_patch_size, padding=0)
#
#         # Remove the fully connected layer
#         self.resnet18.fc = nn.Identity()
#
#         # Add a new fully connected layer for matrix multiplication
#         self.fc = nn.Linear(512, num_classes, bias=True)
#
#     def forward(self, x, Inference=False):
#         x = self.resnet18.conv1(x)
#         print(f'following conv1: {x.shape}')
#         x = self.resnet18.bn1(x)
#         x = self.resnet18.relu(x)
#         x = self.resnet18.maxpool(x)
#         print(f'following maxpool: {x.shape}')
#         x = self.resnet18.layer1(x)
#         print(f'following layer1: {x.shape}')
#         x = self.resnet18.layer2(x)
#         print(f'following layer2: {x.shape}')
#         x = self.resnet18.layer3(x)
#         print(f'following layer3: {x.shape}')
#         x = self.resnet18.layer4(x)
#         print(f'following layer4: {x.shape}')
#         x = self.resnet18.avgpool(x)
#         print(f'following avgpool: {x.shape}')
#
#         if self.training:
#             x = torch.flatten(x, 1)
#             print(f'following flatten: {x.shape}')
#             x = self.fc(x)
#             print(f'following fc: {x.shape}')
#
#         else:
#             x = x.reshape(1, 512, self.grid_size**2).swapaxes(1,2)    # (1, 512, 10, 10) -> (1, 512, 100) -> (1, 100, 512)
#             print(f'following reshape: {x.shape}')
#             x = self.fc(x)                     # (1, 100, 512) * (512, 2) -> (1, 100, 2)
#             print(f'following fc: {x.shape}')
#             x = torch.argmax(x, axis=2)                     # (1, 100, 2) -> (1, 100)
#             print(f'following argmax: {x.shape}')
#             x = x.reshape(1, self.grid_size, self.grid_size)  # (1, 100) -> (1, 10, 10)
#             print(f'following argmax: {x.shape}')
#         return x

class BinaryModifiedResNet18(nn.Module):
    def __init__(self, num_classes=2, grid_size=10, image_size=640):
        super(BinaryModifiedResNet18, self).__init__()
        # Load a pre-trained ResNet-18 model
        self.resnet18 = models.resnet18(pretrained=True)

        self.grid_size = grid_size
        self.patch_size = image_size/grid_size
        self.layer_4_patch_size = int(self.patch_size * (1 / 2 ** 5))
        # print(f"self.layer_4_patch_size: {self.layer_4_patch_size}")

        # Modify the avgpool layer
        self.resnet18.avgpool = nn.AvgPool2d(kernel_size=self.layer_4_patch_size, stride=self.layer_4_patch_size, padding=0)

        # Remove the fully connected layer
        self.resnet18.fc = nn.Identity()

        # Add a new fully connected layer for matrix multiplication
        # self.fc = nn.Linear(512, num_classes, bias=True)
        self.fc = nn.Linear(512, 1, bias=True)

    def forward(self, x, inference=False):
        # print(f'Tensor input: {x.shape}')
        x = self.resnet18.conv1(x)
        # print(f'following conv1: {x.shape}')
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        # print(f'following maxpool: {x.shape}')
        x = self.resnet18.layer1(x)
        # print(f'following layer1: {x.shape}')
        x = self.resnet18.layer2(x)
        # print(f'following layer2: {x.shape}')
        x = self.resnet18.layer3(x)
        # print(f'following layer3: {x.shape}')
        x = self.resnet18.layer4(x)
        # print(f'following layer4: {x.shape}')
        x = self.resnet18.avgpool(x)
        # print(f'following avgpool: {x.shape}')

        if not inference:
            x = torch.flatten(x, 1)
            # print(f'following flatten: {x.shape}')
            x = self.fc(x)
            # print(f'following fc: {x.shape}')

        else:
            x = x.reshape(1, 512, self.grid_size**2).swapaxes(1, 2)    # (1, 512, 10, 10) -> (1, 512, 100) -> (1, 100, 512)
            # print(f'following reshape: {x.shape}')
            x = self.fc(x)                     # (1, 100, 512) * (512, 2) -> (1, 100, 2)
            # print(f'following fc: {x.shape}')
            x = torch.squeeze(x, 2)                     # (1, 100, 2) -> (1, 100)
            # print(f'following squeeze: {x.shape}')
            x = x.reshape(1, self.grid_size, self.grid_size)  # (1, 100) -> (1, 10, 10)
            # print(f'following reshape: {x.shape}')
        x = torch.sigmoid(x)
        return x

# in1 = torch.randn((1, 3, 640, 640))
# in2 = torch.randn((1, 3, 128, 128))
# debug_resnet = BinaryModifiedResNet18(num_classes=2, grid_size=5)
# print(in2.shape)
# res = debug_resnet(in2) # training
# print()
# print(in1.shape)
# debug_resnet.eval()
# res = debug_resnet(in1) # Inference