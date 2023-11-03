import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms

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

# Define the number of epochs
num_epochs = 10

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add other necessary transformations here
])

# Define your dataset
# dataset = BinaryClassificationDataset(root_dir='your_data_directory', transform=transform)

# Split your dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Define dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
