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

class MatchPredictorFCNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[2560, 1280, 640], output_size=3):
        """
        input_size: number of features per game (after preprocessing)
        output_size: number of classes (home win, draw, away win)
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.out = nn.Linear(hidden_sizes[2], output_size)

        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        x = self.out(x)
        return x
    
def train_model(model, train_loader, loss_fn, optimizer):
    model.train()
    losses = []

    for features, labels in train_loader:
        features = features.to('cuda')
        labels = labels.to('cuda')

        optimizer.zero_grad()
        output = model(features)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)
    
def evaluate_model(model, val_loader, loss_fn):
    model.eval()
    losses = []
    accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to('cuda')
            labels = labels.to('cuda')

            output = model(features)
            loss = loss_fn(output, labels)
            losses.append(loss.item())

            _, preds = output.max(1)
            accuracy += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return np.mean(losses), accuracy / total_samples