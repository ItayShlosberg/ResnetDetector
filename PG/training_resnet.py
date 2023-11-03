import os
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

device = 'cpu'
batch_size = 8
num_epochs = 30
DATASET_PATH = r'C:\Users\itay\Desktop\IDF\datasets\patches_coco_val\processed_data'
train_set_size = 0.7
val_set_size = 0.15

class BinaryClassificationDataset(Dataset):
    def __init__(self, root_dir, training_mode, patch_size):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        # Define a transform for data augmentation
        augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Flip the image horizontally with a probability of 0.5
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Apply color jitter
            transforms.RandomResizedCrop(patch_size, scale=(1.0, 1.0), ratio=(1.0, 1.0)),  # Randomly crop a 64x64 patch
            ])

        # Define a transform to normalize the data
        validate_transform = transforms.Compose([
            # transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform = transforms.Compose([augmentation_transform, validate_transform]) if training_mode else validate_transform

        self.negative_files = [os.path.join(root_dir, 'negative', f) for f in
                               os.listdir(os.path.join(root_dir, 'negative'))]
        self.positive_files = [os.path.join(root_dir, 'positive', f) for f in
                               os.listdir(os.path.join(root_dir, 'positive'))]
        self.all_files = self.negative_files + self.positive_files
        self.labels = [0] * len(self.negative_files) + [1] * len(self.positive_files)


    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_name = self.all_files[idx]
        image = Image.open(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

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
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    # Now you can iterate over the dataloader to get batches of images and labels
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        print(f'# of label 1: {sum(labels)}')
        print(f'# of label 0: {sum(labels == 0)}')

def create_datasets():
    dataset = BinaryClassificationDataset(root_dir=DATASET_PATH,
                                          training_mode=True,
                                          patch_size=128)
    train_size = int(train_set_size * len(dataset))
    val_size = int(val_set_size * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return dataset, train_dataset, val_dataset, test_dataset


def create_dataloader(train_dataset, val_dataset, test_dataset):
    # Define dataloaders
    # train_sampler = create_sampler_for_balanced_classes(train_dataset.labels)
    # val_sampler = create_sampler_for_balanced_classes(val_dataset.labels)
    # test_sampler = create_sampler_for_balanced_classes(test_dataset.labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader, test_loader

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


if __name__ == '__main__':

    # Create the dataset
    dataset, train_dataset, val_dataset, test_dataset = create_datasets()
    train_loader, val_loader, test_loader = create_dataloader(train_dataset, val_dataset, test_dataset)


    # Load a pre-trained ResNet18 model and modify for binary classification
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Change the fully connected layer

    # Send the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Use Binary Cross Entropy with Logits as the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Use Stochastic Gradient Descent as the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Instantiate early stopping
    early_stopping = EarlyStopping(patience=7, verbose=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1).float()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')


    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    print(f'Validation Loss: {val_loss / len(val_loader)}')

    # Test loop
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    print(f'Test Loss: {test_loss / len(test_loader)}')


    DEBUG = 1
