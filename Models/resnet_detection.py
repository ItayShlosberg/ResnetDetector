import torch
import torch.nn as nn
import torchvision.models as models



class Resnet18Detector(nn.Module):
    # For the binary case with one class
    def __init__(self, image_size=640, pretrained=True):
        super(Resnet18Detector, self).__init__()

        # Check if image_size is divisible by 32 (2**5)
        assert image_size % 32 == 0, "image_size should be divisible by 32 due to the shape of the feature tensor following layer 4 of ResNet after 5 layers with stride 2."

        self.image_size = image_size

        # Load a pre-trained ResNet-18 model
        self.resnet18 = models.resnet18(pretrained=pretrained)

        # Remove the fully connected layer
        self.resnet18.fc = nn.Identity()

        # Add a new fully connected layer for matrix multiplication
        # self.fc = nn.Linear(512, num_classes, bias=True) # For non-binary case replace with this line.
        self.fc = nn.Linear(512, 1, bias=True)


    def forward(self, x):
        # Check if the input X has dimensions that can be divided evenly by image_size
        assert self.image_size % x.shape[-2] == 0 and \
               self.image_size % x.shape[-1] == 0, f"Input dimensions {x.shape[-2]}x{x.shape[-1]} cannot be divided evenly by image_size {self.image_size}."

        # print(f'Tensor input: {x.shape}')
        x = self.resnet18(x)
        # print(f'following Resnet layers: {x.shape}')
        x = self.fc(x)
        # print(f'following fc: {x.shape}')
        x = torch.sigmoid(x)

        return x



# in1 = torch.randn((1, 3, 640, 640))
# in2 = torch.randn((1, 3, 128, 128))
# debug_resnet = Resnet18Detector(image_size=640)
# print(in2.shape)
# res = debug_resnet(in2) # training
# print()
# print(in1.shape)
# debug_resnet.eval()
# res = debug_resnet(in1) # Inference