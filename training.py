import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import dataHandler
from model import MatchPredictorFCNN, evaluate_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import fbrefdata as fd
import json


def train_model(config : dict,  
                df_fifa: pd.DataFrame,
                df_lineups: pd.DataFrame,
                df_matches: pd.DataFrame,
                df_players: pd.DataFrame,
                df_teams: pd.DataFrame,
                df_competitions: pd.DataFrame):
   
    features = pd.DataFrame()
    labels = torch.tensor([])
    i = 0
    for _, game in df_matches.iterrows():
        frame = dataHandler.evaluate_two_teams_by_game_id(game['game_id'], df_matches, df_players, df_fifa, df_lineups)

        features = pd.concat([features, frame], ignore_index=True)
        home_club_goals = game['home_club_goals']
        away_club_goals = game['away_club_goals']

        if (home_club_goals > away_club_goals):
            labels.add(0)
        elif (home_club_goals == away_club_goals):
            labels.add(1)
        else:
            labels.add(2)
        i += 1

        if  (i > 100):
            break

    features = features.drop(['team1_game_team_id', 'team2_game_team_id'], axis=1)
    #labels = torch.randint(0, 3, (1000,))  # Random labels (0, 1, or 2)

    # Create DataLoader for training and validation sets
    dataset = TensorDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = MatchPredictorFCNN(input_size=config["input_size"])
    loss_fn = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(config["number_of_epochs"]):
        train_loss = train_model(model, train_loader, loss_fn, optimizer)
        val_loss, val_accuracy = evaluate_model(model, val_loader, loss_fn)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
