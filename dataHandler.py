import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from model import MatchPredictorFCNN, evaluate_model, train_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import fbrefdata as fd

def get_players_evaluation_df(df_players, df_fifa):
    """
    Merges player data with FIFA data, selects relevant columns,
    calculates a custom player evaluation, and returns the resulting DataFrame.

    Args:
        df_players: DataFrame containing player data.
        df_fifa: DataFrame containing FIFA data.

    Returns:
        DataFrame: DataFrame with selected columns and custom player evaluation.
    """

    # Merge the DataFrames
    merged_players_fifa_df = pd.merge(df_players, df_fifa, left_on='name', right_on='full_name', how='inner')

    # Select relevant columns
    selected_columns = ['player_id', 'position', 'full_name', 'current_club_id', 'overall_rating', 'potential', 'value_euro', 'game_team_id']
    players_evaluation_df = merged_players_fifa_df[selected_columns].copy()

    # Define columns for custom evaluation
    evaluated_columns = ['crossing', 'finishing', 'heading_accuracy', 'short_passing', 'volleys', 'dribbling', 'curve',
                        'freekick_accuracy', 'long_passing', 'ball_control', 'acceleration', 'sprint_speed', 'agility',
                        'reactions', 'balance', 'shot_power', 'stamina', 'strength', 'aggression', 'positioning',
                        'penalties']

    # Calculate custom player evaluation
    players_evaluation_df.loc[:, 'custom_player_evaluation'] = merged_players_fifa_df[evaluated_columns].mean(axis=1)

    return players_evaluation_df

########################


def calculate_position_stats(merged_teams_players, position, stats_columns):
    """
    Calculates average stats for a specific position.

    Args:
        merged_teams_players: DataFrame containing merged data for teams and players.
        position: The position to filter for (e.g., 'Defender', 'Goalkeeper', 'Attacker').
        stats_columns: List of columns to calculate averages for (e.g., ['overall_rating', 'potential']).

    Returns:
        DataFrame: DataFrame with average stats for the specified position.
    """

    # Filter for the specified position
    position_df = merged_teams_players[merged_teams_players['position'] == position]


    # Group by current_club_id and calculate averages
    position_stats = position_df.groupby('game_team_id')[stats_columns].mean().reset_index()

    # Rename columns
    prefix = position.lower() + '_'
    position_stats = position_stats.rename(columns={col: prefix + col for col in stats_columns})

    return position_stats

def evaluate_two_player_dfs(df_players1, df_players2, df_fifa):
    """Evaluates two player DataFrames and returns a single DataFrame with position stats.

    Args:
        df_players1: DataFrame containing player data for team 1.
        df_players2: DataFrame containing player data for team 2.
        df_fifa: DataFrame containing FIFA player ratings.

    Returns:
        DataFrame with position stats for both teams.
    """

    players_1_evaluation_df = get_players_evaluation_df(df_players1, df_fifa)
    players_2_evaluation_df = get_players_evaluation_df(df_players2, df_fifa)

  # Define positions and stats columns
    positions = ['Defender', 'Goalkeeper', 'Attack', 'Midfield']
    stats_columns = ['overall_rating', 'potential']

    # Calculate position stats for both teams
    stats_players1 = {}
    stats_players2 = {}
    for position in positions:
        stats_players1[position] = calculate_position_stats(players_1_evaluation_df, position, stats_columns)
        stats_players2[position] = calculate_position_stats(players_2_evaluation_df, position, stats_columns)

    

    df_result = pd.concat([
        pd.concat(list(stats_players1.values()), axis=1).add_prefix('team1_'),  # Add prefix for players1
        pd.concat(list(stats_players2.values()), axis=1).add_prefix('team2_')  # Add prefix for players2
    ], axis=1)

    return df_result



def evaluate_two_teams_by_game_id(game_id):

  #hra
  df_matches_filtered = df_matches[df_matches['game_id'] == game_id]
  home_club_id = df_matches_filtered['home_club_id'].iloc[0]  
  away_club_id = df_matches_filtered['away_club_id'].iloc[0] 

  #ziskanie hracov z lineups pre konkr. hru

  df_players_1_IDs_of_game = df_lineups.loc[
      (df_lineups['game_id'] == game_id) & (df_lineups['club_id'] == home_club_id),
      'player_id'
  ]
  #ziskanie hracov z lineups pre konkr. hru

  df_players_2_IDs_of_game = df_lineups.loc[
    (df_lineups['game_id'] == game_id) & (df_lineups['club_id'] == away_club_id),
    'player_id'
  ]

  df_players_1 = df_players[df_players['player_id'].isin(df_players_1_IDs_of_game)].copy()
  df_players_2 = df_players[df_players['player_id'].isin(df_players_2_IDs_of_game)].copy()

  #pridanie stlpca hry pre groupovanie
  df_players_1['game_team_id'] = home_club_id 
  df_players_2['game_team_id'] = away_club_id

  df_evaluated = evaluate_two_player_dfs(df_players_1, df_players_2, df_fifa)

  return df_evaluated




#momentalne nefunguje
def evaluate_teams(df_teams, df_players, df_fifa):
    """
    Evaluates teams based on average player stats for different positions.

    Args:
        df_teams: DataFrame containing team data.
        df_players: DataFrame containing player data.
        df_fifa: DataFrame containing FIFA data.

    Returns:
        DataFrame: DataFrame with team evaluations.
    """

    # Get player evaluations
    players_evaluation_df = get_players_evaluation_df(df_players, df_fifa)  # Assuming evaluate_players is defined elsewhere

    # Merge team and player data
    merged_teams_players = pd.merge(df_teams, players_evaluation_df, left_on='club_id', right_on='current_club_id', how='inner')

    # Calculate stats for each position
    defender_stats = calculate_position_stats(merged_teams_players, 'Defender', ['overall_rating', 'potential'])
    goalkeeper_stats = calculate_position_stats(merged_teams_players, 'Goalkeeper', ['overall_rating', 'potential'])
    attacker_stats = calculate_position_stats(merged_teams_players, 'Attack', ['overall_rating', 'potential'])

    # Merge stats with team data
    merged_teams_defender_stats_df = pd.merge(df_teams, defender_stats, left_on='club_id', right_on='current_club_id', how='left')
    merged_teams_all_stats_df = pd.merge(merged_teams_defender_stats_df, goalkeeper_stats, left_on='club_id', right_on='current_club_id', how='left')
    final_df = pd.merge(merged_teams_all_stats_df, attacker_stats, left_on='club_id', right_on='current_club_id', how='left')

    # Select desired columns
    selected_columns = ['club_id', 'name', 'club_code', 'squad_size', 'defender_overall_rating', 'defender_potential',
                        'goalkeeper_overall_rating', 'goalkeeper_potential', 'attack_overall_rating', 'attack_potential']
    extracted_df = final_df[selected_columns]

    return extracted_df