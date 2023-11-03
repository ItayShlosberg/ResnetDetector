from Models.modified_resnet import *
import os
import numpy as np
import matplotlib.pyplot as plt
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
from os.path import join
import torchvision.transforms.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR


DEVICE = 'cuda'
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

DATASET_PATH = r'C:\Users\itay\Desktop\IDF\datasets\patches_coco_val\EXP\TEST1'
OUTPUT_FOLDER = r'C:\Users\itay\Desktop\IDF\models\train\3.11.23'

train_set_size = 1#0.95
val_set_size = 0 #.025

def create_sampler_for_balanced_classes(labels):
    weight_per_class = [len(labels)/sum([not xx for xx in labels]), len(labels)/sum([xx for xx in labels])]
    weights = [weight_per_class[i] for i in labels]
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler

# TODO: remove
def random_dataloader():
    # Create the dataloader
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    #########

    # For unbalanced dataset we create a weighted sampler
    sampler = create_sampler_for_balanced_classes(dataset.labels)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    # Now you can iterate over the dataloader to get batches of images and labels
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        print(f'# of label 1: {sum(labels)}')
        print(f'# of label 0: {sum(labels == 0)}')

def create_datasets():
    dataset = PatchesDataset(root_dir=DATASET_PATH,
                             training_mode=True,
                             patch_size=128)
    train_size = int(train_set_size * len(dataset))
    val_size = int(val_set_size * len(dataset))
    test_size = len(dataset) - train_size - val_size
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataset = dataset
    val_dataset = dataset
    test_dataset = dataset
    return dataset, train_dataset, val_dataset, test_dataset


def create_dataloader(train_dataset, val_dataset, test_dataset):
    # Define dataloaders
    train_sampler = create_sampler_for_balanced_classes(train_dataset.labels)
    # val_sampler = create_sampler_for_balanced_classes(val_dataset.labels)
    # test_sampler = create_sampler_for_balanced_classes(test_dataset.labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader



if __name__ == '__main__':

    # Create the dataset
    dataset, train_dataset, val_dataset, test_dataset = create_datasets()
    train_loader, val_loader, test_loader = create_dataloader(train_dataset, val_dataset, test_dataset)


    # Load a pre-trained ResNet18 model and modify for binary classification
    # model = models.resnet18(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, 1)  # Change the fully connected layer
    model = BinaryModifiedResNet18(num_classes=2, grid_size=5)


    # Send the model to GPU if available
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)

    # Use Binary Cross Entropy with Logits as the loss function
    # criterion = nn.BCEWithLogitsLoss() # In cases the sigmoid is NOT the last layer of the modified Resnet model
    # use Binary Cross Entropy as the loss function
    criterion = nn.BCELoss() # In cases the sigmoid is the last layer of the modified Resnet model

    # Use Stochastic Gradient Descent as the optimizer
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Instantiate early stopping
    # early_stopping = EarlyStopping(patience=7, verbose=True)

    # Initialize LR scheduler
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    import torch.optim.lr_scheduler as lr_scheduler
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=100)

    losses = []
    # Training loop
    for epoch in range(NUM_EPOCHS):
        if epoch%100==0:
            DEBUG=0
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1).float()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(f'labels: {labels.flatten().to("cpu").detach().numpy()}')
            print(f'outputs: {outputs.flatten().to("cpu").detach().numpy().round(2)}\n')
            # Backward pass and optimize
            loss.backward()
            optimizer.step()


            # Print statistics
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
        # print(f'LR: {optimizer.param_groups[0]["lr"]}')
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: SGD lr %.6f -> %.6f" % (epoch, before_lr, after_lr))

        losses.append(running_loss / len(train_loader))



        plt.plot(np.arange(len(losses)), losses)
        plt.savefig(join(OUTPUT_FOLDER, r'losses.jpg'))


        # # Validation loop
    # model.eval()
    # val_loss = 0.0
    # with torch.no_grad():
    #     for inputs, labels in val_loader:
    #         inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1).float()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         val_loss += loss.item()
    # print(f'Validation Loss: {val_loss / len(val_loader)}')
    #
    # # Test loop
    # test_loss = 0.0
    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1).float()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         test_loss += loss.item()
    # print(f'Test Loss: {test_loss / len(test_loader)}')

    model.to('cpu')
    torch.save(model.state_dict(), join(OUTPUT_FOLDER, f'checkpoint_ep{NUM_EPOCHS}.pt'))
    print(f'Model saved at: {join(OUTPUT_FOLDER, f"checkpoint_ep{NUM_EPOCHS}.pt")}')
    DEBUG = 1