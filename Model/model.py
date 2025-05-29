import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import fbrefdata as fd
from sklearn.metrics import confusion_matrix

class MatchPredictorFCNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32, 16], output_size=4):
        """
        input_size: number of features per game (after preprocessing)
        output_size: number of classes (home win, draw, away win)
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.out = nn.Linear(hidden_sizes[3], output_size)

        self.dropout = nn.Dropout(p=0.1)
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)

        x = self.out(x)
        return x
    
def train_model(model, train_loader, loss_fn, optimizer):
    model.train()
    losses = []
    accuracy = 0
    total_samples = 0

    for features, labels in train_loader:
        features = features.to('cpu')
        labels = labels.to('cpu')

        optimizer.zero_grad()
        output = model(features)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        _, preds = output.max(1)
        accuracy += (preds == labels).sum().item()
        total_samples += labels.size(0)
        losses.append(loss.item())

    return np.mean(losses), accuracy / total_samples
    
# def evaluate_model(model, val_loader, loss_fn):
#     model.eval()
#     losses = []
#     accuracy = 0
#     total_samples = 0
#
#     with torch.no_grad():
#         for features, labels in val_loader:
#             features = features.to('cpu')
#             labels = labels.to('cpu')
#
#             output = model(features)
#             loss = loss_fn(output, labels)
#             losses.append(loss.item())
#
#             _, preds = output.max(1)
#             accuracy += (preds == labels).sum().item()
#             total_samples += labels.size(0)

    return np.mean(losses), accuracy / total_samples

def evaluate_model(model, val_loader, loss_fn):
    model.eval()
    losses = []
    accuracy = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to('cpu')
            labels = labels.to('cpu')

            output = model(features)
            loss = loss_fn(output, labels)
            losses.append(loss.item())

            _, preds = output.max(1)
            accuracy += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # Collect predictions and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute mean loss and accuracy
    mean_loss = np.mean(losses)
    mean_accuracy = accuracy / total_samples

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Home Win', 'Draw', 'Away Win'],
                yticklabels=['Home Win', 'Draw', 'Away Win'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return mean_loss, mean_accuracy