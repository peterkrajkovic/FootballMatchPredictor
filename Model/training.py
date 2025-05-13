import pandas as pd
from Features.match_features import get_dataframe_game_id
from sklearn.preprocessing import StandardScaler
from Features.team_features import get_average_goals_conceded, get_average_goals_scored, get_clean_sheet_rate, get_current_league_points, get_days_rest, get_form_points, get_goal_difference_momentum, get_mutual_goal_difference, get_mutual_statistic, get_result_rate, get_team_points
from Model.model import MatchPredictorFCNN, evaluate_model, train_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from graphs import show_training_progress

def trainModel(config : dict,  
                df_fifa: pd.DataFrame,
                df_lineups: pd.DataFrame,
                df_matches: pd.DataFrame,
                df_players: pd.DataFrame,
                df_teams: pd.DataFrame,
                df_competitions: pd.DataFrame):
   
    features = pd.DataFrame()
    labels = []
    # Filter only Premier League matches (assuming "GB1" is the ID for that)
    df_matches["date"] = pd.to_datetime(df_matches["date"], dayfirst=True, errors='coerce')
    df_matches = df_matches[
        (df_matches["competition_id"] == "GB1") & 
        (df_matches["date"] > "2015-10-28")
    ]

    for i, (_, game) in enumerate(df_matches.iterrows()):
        game_id = game["game_id"]
        frame = get_dataframe_game_id(game["game_id"], df_matches, df_players, df_fifa, df_lineups)

        if frame is not None and not frame.empty:
            num_rows = len(frame)
            features = pd.concat([features, frame], ignore_index=True)

            home_goals = game['home_club_goals']
            away_goals = game['away_club_goals']
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

            labels.extend([label] * num_rows)  # One label for each row of features

        if i > 1000:
            break
    
    print("Number of features:", features.shape[1])
    print("All feature columns:")
    print(features.columns.tolist())
    features = features.fillna(0)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Create DataLoader for training and validation sets
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), 
                       torch.tensor(labels, dtype=torch.long))    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = MatchPredictorFCNN(input_size=features.shape[1])
    loss_fn = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    bestAccuracy = config["best_accuracy"]
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    for epoch in range(config["number_of_epochs"]):
        train_loss, train_accuracy = train_model(model, train_loader, loss_fn, optimizer)
        test_loss,test_accuracy = evaluate_model(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f},Train accuracy: {train_accuracy:.4f} Test Loss: {test_loss:.4f}, Test Accuracy: {(test_accuracy * 100):.4f}%")
        if (test_accuracy > bestAccuracy):
            torch.save(model.state_dict(), config["model_path"])
            config["best_accuracy"] = test_accuracy
            bestAccuracy = test_accuracy

    print("best accuracy :")
    print(bestAccuracy)
    epochs = range(1, config["number_of_epochs"] + 1)
    show_training_progress(epochs, train_losses, test_losses, train_accuracies, test_accuracies)