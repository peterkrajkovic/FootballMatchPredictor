import pandas as pd
from sklearn.preprocessing import StandardScaler
from Model.model import MatchPredictorFCNN, evaluate_model, train_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from graphs import show_training_progress
import numpy as np

import utils


def prepareModel(config: dict,
               dataset: pd.DataFrame):
    features = pd.DataFrame()
    labels = []

    for i, (_, game) in enumerate(dataset.iterrows()):

        if game is not None and not game.empty:

            home_goals = game.home_goals
            away_goals = game.away_goals

            game.drop(columns=["home_goals", "away_goals"], inplace=True)

            # num_rows = len(game)
            # player_columns = [
            #     "team1_defender_overall_rating", "team1_defender_potential", "team1_defender_market_value_in_eur",
            #     "team1_goalkeeper_overall_rating", "team1_goalkeeper_potential", "team1_goalkeeper_market_value_in_eur",
            #     "team1_attack_overall_rating", "team1_attack_potential", "team1_attack_market_value_in_eur",
            #     "team1_midfield_overall_rating", "team1_midfield_potential", "team1_midfield_market_value_in_eur",
            #     "team2_defender_overall_rating", "team2_defender_potential", "team2_defender_market_value_in_eur",
            #     "team2_goalkeeper_overall_rating", "team2_goalkeeper_potential", "team2_goalkeeper_market_value_in_eur",
            #     "team2_attack_overall_rating", "team2_attack_potential", "team2_attack_market_value_in_eur",
            #     "team2_midfield_overall_rating", "team2_midfield_potential", "team2_midfield_market_value_in_eur"
            # ]

            all_columns = [
                    "team1_defender_overall_rating",
                    "team1_defender_potential",
                    "team1_defender_market_value_in_eur",
                    "team1_goalkeeper_overall_rating",
                    "team1_goalkeeper_potential",
                    "team1_goalkeeper_market_value_in_eur",
                    "team1_attack_overall_rating",
                    "team1_attack_potential",
                    "team1_attack_market_value_in_eur",
                    "team1_midfield_overall_rating",
                    "team1_midfield_potential",
                    "team1_midfield_market_value_in_eur",
                    "team2_defender_overall_rating",
                    "team2_defender_potential",
                    "team2_defender_market_value_in_eur",
                    "team2_goalkeeper_overall_rating",
                    "team2_goalkeeper_potential",
                    "team2_goalkeeper_market_value_in_eur",
                    "team2_attack_overall_rating",
                    "team2_attack_potential",
                    "team2_attack_market_value_in_eur",
                    "team2_midfield_overall_rating",
                    "team2_midfield_potential",
                    "team2_midfield_market_value_in_eur",
                    "home_form",
                    "away_form",
                    "home_win_rate",
                    "home_draw_rate",
                    "away_win_rate",
                    "away_draw_rate"
            ]



            #df_players_only = game[player_columns].to_frame().T
            df_all = game[all_columns].to_frame().T
            features = pd.concat([features, df_all], ignore_index=True)
            # features = pd.concat([features, game], ignore_index=True)
            """try:
                home_points, away_points = get_team_points(df_matches, game_id)
                home_form, away_form = get_form_points(df_matches, game_id, form_n=10)
                h_wr, h_dr, a_wr, a_dr = get_result_rate(df_matches, game_id)
                h2h_home, h2h_away = get_mutual_statistic(df_matches, game_id)
                home_rest_days, away_rest_days = get_days_rest(df_matches, game_id)
                home_scored, away_scored = get_average_goals_scored(df_matches, game_id)
                home_conceded, away_conceded = get_average_goals_conceded(df_matches, game_id)
                home_diff, away_diff = get_mutual_goal_difference(df_matches, game_id)
                home_raw_points, away_raw_points = get_current_league_points(df_matches, game_id)
                home_momentum, away_momentum = get_goal_difference_momentum(df_matches, game_id)
                home_clean_sheet, away_clean_sheet = get_clean_sheet_rate(df_matches, game_id)
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

            extra_columns = [
                'home_points', 'away_points',
                'home_form', 'away_form',
                'h_wr', 'h_dr', 'a_wr', 'a_dr',
                'h2h_home', 'h2h_away',
                'home_rest_days', 'away_rest_days',
                'home_scored', 'away_scored',
                'home_conceded', 'away_conceded',
                'home_diff', 'away_diff',
                'home_raw_points', 'away_raw_points',
                'home_momentum', 'away_momentum',
                'home_clean_sheet', 'away_clean_sheet'
            ]

            # 2. Create DataFrame from feature_row
            extra_df = pd.DataFrame([feature_row], columns=extra_columns)

            # 3. Concatenate horizontally with frame
            combined = pd.concat([frame.reset_index(drop=True), extra_df], axis=1)

            # 4. Add to features
            features = pd.concat([features, combined], ignore_index=True)
            """
            if home_goals > away_goals:
                label = 0
            elif home_goals == away_goals:
                label = 1
            else:
                label = 2

            labels.extend([label] * 1)  # One label for each row of features

    features.to_csv('features_all.csv', index=False)
    with open('labels_all.txt', 'w') as f:
        for l in labels:
            f.write(f"{l}\n")

def trainModel(config: dict, features: pd.DataFrame, labels: list[float]):

    """print("Number of features:", features.shape[1])
    print("All feature columns:")
    print(features.columns.tolist())
    features.to_csv('Data/raw_features.csv',index=False,sep=';')
    features = features.fillna(0)
    features.to_csv('Data/filled_nan_features.csv', index=False,sep=';')
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    df = pd.DataFrame(features)
    df.to_csv("Data/scaled_features.csv", index=False,sep=';')"""
    features = features.fillna(0)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    df = pd.DataFrame(features)

    # Create DataLoader for training and validation sets
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32),
                            torch.tensor(labels, dtype=torch.long))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

    model = MatchPredictorFCNN(input_size=features.shape[1])


    loss_fn = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"],weight_decay=1e-5 )
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    #optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)




    # Check if GPU is available
    device = torch.device(utils.selectGPU())
    model.to(device)

    bestAccuracy = config["best_accuracy"]
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    for epoch in range(config["number_of_epochs"]):
        train_loss, train_accuracy = train_model(model, train_loader, loss_fn, optimizer)
        test_loss, test_accuracy = evaluate_model(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        #pridane scheduler
        #scheduler.step(test_loss)
        print(
            f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f},Train accuracy: {train_accuracy:.4f} Test Loss: {test_loss:.4f}, Test Accuracy: {(test_accuracy * 100):.4f}%")
        if (test_accuracy > bestAccuracy):
            torch.save(model.state_dict(), config["model_path"])
            config["best_accuracy"] = test_accuracy
            bestAccuracy = test_accuracy

    print("best accuracy :")
    print(bestAccuracy)
    epochs = range(1, config["number_of_epochs"] + 1)
    show_training_progress(epochs, train_losses, test_losses, train_accuracies, test_accuracies)