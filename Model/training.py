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
                dataset: pd.DataFrame):
   
    features = []
    labels = []

    for i, (_, game) in enumerate(dataset.iterrows()):
        if game is not None and not game.empty:
            home_goals = game['home_goals']
            away_goals = game['away_goals']

            game = game.drop(labels=['home_goals', 'away_goals'])

            if home_goals > away_goals:
                label = 0
            elif home_goals == away_goals:
                label = 1
            else:
                label = 2

            features.append(game.values)  # Save the row (as a list of values)
            labels.append(label)

        if i > 1000:
            break

    # Convert to proper DataFrame and Series
    features_df = pd.DataFrame(features, columns=dataset.drop(columns=['home_goals', 'away_goals']).columns)
    labels_series = pd.Series(labels)
    
    print("Number of features:", features_df.shape[1])
    print("All feature columns:")
    print(features_df.columns.tolist())
    features_df = features_df.fillna(0)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Create DataLoader for training and validation sets
    features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_series.values, dtype=torch.long)

    # Wrap into a TensorDataset
    dataset = TensorDataset(features_tensor, labels_tensor)

    # Split into train and validation sets
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