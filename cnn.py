import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve


# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 92, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(92, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(2 * 128 * 128, 92),
            nn.ReLU(),
            nn.Linear(92, 48),
            nn.ReLU(),
            nn.Linear(48, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# Set random seed for reproducibility
torch.manual_seed(0)

# Define hyperparameters
batch_size = 30
num_epochs = 50
learning_rate = 0.001
train_ratio = 0.7
num_folds = 5

# Load the DICOM dataset and normalize the images
data_path = '/path/to/dicom/folder'
images = []
labels = []

for filename in os.listdir(data_path):
    if filename.endswith('.dcm'):
        image = pydicom.dcmread(os.path.join(data_path, filename)).pixel_array
        image = np.float32(image) / np.max(image)
        images.append(image)
        labels.append(int('fat' in filename))

images = np.array(images)
labels = np.array(labels)

# Split the dataset into 70% training and 30% testing sets
train_dataset, test_dataset = random_split(dataset, [int(train_ratio * len(dataset)), len(dataset) - int(train_ratio * len(dataset))])

# K-fold cross-validation on training set
kf = KFold(n_splits=num_folds, shuffle=True)
fold_train_accuracy = []
fold_test_accuracy = []
fold_train_loss = []
fold_test_loss = []

for fold, (train_index, _) in enumerate(kf.split(train_dataset)):
    # Create data loaders for batch processing
    train_subset = torch.utils.data.Subset(train_dataset, train_index)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader_fold = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    # Initialize the CNN model
    model = CNNModel()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_accuracy = []
    test_accuracy = []
    train_loss = []
    test_loss = []

    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0
        epoch_train_loss = 0.0
        epoch_test_loss = 0.0

        # Training
        model.train()
        for images, labels in train_loader_fold:
            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_accuracy.append(100.0 * train_correct / train_total)
        train_loss.append(epoch_train_loss / len(train_loader_fold))

        # Testing
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                epoch_test_loss += loss.item()

        test_accuracy.append(100.0 * test_correct / test_total)
        test_loss.append(epoch_test_loss / len(test_loader))
        
    # Evaluation metrics
    model.eval()
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_predictions.extend(predicted.tolist())
            test_labels.extend(labels.tolist())

    test_accuracy = accuracy_score(test_labels, test_predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_predictions).ravel()
    test_specificity = tn / (tn + fp)
    test_sensitivity = tp / (tp + fn)
    test_auc = roc_auc_score(test_labels, test_predictions)
    precision, recall, _ = precision_recall_curve(test_labels, test_predictions)
    test_prc = auc(recall, precision)

    print(f"Fold {fold+1}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Specificity: {test_specificity}")
    print(f"Test Sensitivity: {test_sensitivity}")
    print(f"Test AUC: {test_auc}")
    print(f"Test PRC: {test_prc}")
    print("")

    fold_train_accuracy.append(train_accuracy)
    fold_test_accuracy.append(test_accuracy)
    fold_train_loss.append(train_loss)
    fold_test_loss.append(test_loss)

# Print average test metrics across all folds
print("Average Test Metrics:")
print(f"Accuracy: {np.mean(fold_test_accuracy)}")
print(f"Specificity: {np.mean(fold_test_specificity)}")
print(f"Sensitivity: {np.mean(fold_test_sensitivity)}")
print(f"AUC: {np.mean(fold_test_auc)}")
print(f"PRC: {np.mean(fold_test_prc)}")
