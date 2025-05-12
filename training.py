import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import dataHandler
from model import MatchPredictorFCNN, evaluate_model, train_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import fbrefdata as fd
import json
import torch_directml
device = torch_directml.device()


# def trainModel(config : dict,
#                 df_fifa: pd.DataFrame,
#                 df_lineups: pd.DataFrame,
#                 df_matches: pd.DataFrame,
#                 df_players: pd.DataFrame,
#                 df_teams: pd.DataFrame,
#                 df_competitions: pd.DataFrame):
#
#     features = pd.DataFrame()
#     labels = []
#     # Filter only Premier League matches (assuming "GB1" is the ID for that)
#     df_matches["date"] = pd.to_datetime(df_matches["date"], dayfirst=True, errors='coerce')
#     df_matches = df_matches[
#         (df_matches["competition_id"] == "GB1") &
#         (df_matches["date"] > "2015-10-28")
#     ]
#
#     for i, (_, game) in enumerate(df_matches.iterrows()):
#         frame = dataHandler.evaluate_two_teams_by_game_id(
#             game['game_id'], df_matches, df_players, df_fifa, df_lineups
#             )
#         if frame is not None and not frame.empty:
#             num_rows = len(frame)
#             features = pd.concat([features, frame], ignore_index=True)
#
#             home_goals = game['home_club_goals']
#             away_goals = game['away_club_goals']
#
#             if home_goals > away_goals:
#                   label = 0
#             elif home_goals == away_goals:
#                 label = 1
#             else:
#                 label = 2
#
#             labels.extend([label] * num_rows)  # One label for each row of features
#
#         if i > 10000:
#             break
#
#     features = features.drop(['team1_game_team_id', 'team2_game_team_id'], axis=1)
#     print("Number of features:", features.shape[1])
#     features = features.fillna(0)
#
#     # Create DataLoader for training and validation sets
#     dataset = TensorDataset(torch.tensor(features.values, dtype=torch.float32), torch.tensor(labels))
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
#
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
#
#     model = MatchPredictorFCNN(input_size=config["input_size"])
#     loss_fn = nn.CrossEntropyLoss()  # For multi-class classification
#     optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
#
#     # Check if GPU is available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#
#     bestAccuracy = config["best_accuracy"]
#     for epoch in range(config["number_of_epochs"]):
#         train_loss = train_model(model, train_loader, loss_fn, optimizer)
#         val_loss, val_accuracy = evaluate_model(model, val_loader, loss_fn)
#
#         print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
#         if (val_accuracy > bestAccuracy):
#             torch.save(model.state_dict(), config["model_path"])
#             config["best_accuracy"] = val_accuracy
#             bestAccuracy = val_accuracy
#
#     print("best accuracy :")
#     print(bestAccuracy)

def trainModel(config: dict,
               df_fifa: pd.DataFrame,
               df_lineups: pd.DataFrame,
               df_matches: pd.DataFrame,
               df_players: pd.DataFrame,
               df_teams: pd.DataFrame,
               df_competitions: pd.DataFrame):
    features = []
    labels = []

    df_matches["date"] = pd.to_datetime(df_matches["date"], dayfirst=True, errors='coerce')
    df_matches = df_matches[
        (df_matches["competition_id"] == "GB1") &
        (df_matches["date"] > "2015-10-28")
        ]

    for i, (_, game) in enumerate(df_matches.iterrows()):
        game_id = game['game_id']

        try:
            home_points, away_points = dataHandler.get_team_points(df_matches, game_id)
            home_form, away_form = dataHandler.get_form_points(df_matches, game_id, form_n=10)
            h_wr, h_dr, a_wr, a_dr = dataHandler.get_result_rate(df_matches, game_id)
            h2h_home, h2h_away = dataHandler.get_mutual_statistic(df_matches, game_id)
            home_rest_days, away_rest_days = dataHandler.get_days_rest(df_matches, game_id)
            home_scored, away_scored = dataHandler.get_average_goals_scored(df_matches, game_id)
            home_conceded, away_conceded = dataHandler.get_average_goals_conceded(df_matches, game_id)
            home_diff, away_diff = dataHandler.get_mutual_goal_difference(df_matches, game_id)
            home_raw_points, away_raw_points = dataHandler.get_current_league_points(df_matches, game_id)
            home_momentum, away_momentum = dataHandler.get_goal_difference_momentum(df_matches, game_id)
            home_clean_sheet, away_clean_sheet = dataHandler.get_clean_sheet_rate(df_matches, game_id)
        except Exception as e:
            print(f"Chyba pri spracovaní zápasu {game_id}: {e}")
            continue

        feature_row = [
            home_points, away_points,
            home_form, away_form,
            h_wr, h_dr, a_wr, a_dr,
            h2h_home, h2h_away,
            home_rest_days, away_rest_days,
            home_scored, away_scored,
            home_conceded, away_conceded,
            home_diff, away_diff,
            home_raw_points, away_raw_points,
            home_momentum, away_momentum,
            home_clean_sheet, away_clean_sheet
        ]

        features.append(feature_row)

        home_goals = game['home_club_goals']
        away_goals = game['away_club_goals']

        if home_goals > away_goals:
            label = 0  # Výhra domáceho
        elif home_goals == away_goals:
            label = 1  # Remíza
        else:
            label = 2  # Výhra hosťov

        labels.append(label)

        if i > 10000:  # Obmedzenie pre testovanie
            break


    feature_columns = [
        'home_points', 'away_points',
        'home_form', 'away_form',
        'h_win_rate', 'h_draw_rate', 'a_win_rate', 'a_draw_rate',
        'h2h_home', 'h2h_away',
        'home_rest_days', 'away_rest_days',
        'home_scored', 'away_scored',
        'home_conceded', 'away_conceded',
        'home_diff', 'away_diff',
        'home_raw_points', 'away_raw_points',
        'home_momentum', 'away_momentum',
        'home_clean_sheet', 'away_clean_sheet'
    ]
    features_df = pd.DataFrame(features, columns=feature_columns)
    features_df = features_df.fillna(0)

    dataset = TensorDataset(
        torch.tensor(features_df.values, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


    model = MatchPredictorFCNN(config["input_size"])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_accuracy = 0
    for epoch in range(config["number_of_epochs"]):
        train_loss = train_model(model, train_loader, loss_fn, optimizer)

        val_loss, val_accuracy = evaluate_model(model, val_loader, loss_fn)
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            torch.save(model.state_dict(), config["model_path"])
            best_accuracy = val_accuracy

    print(f"Najlepšia presnosť: {best_accuracy:.4f}")