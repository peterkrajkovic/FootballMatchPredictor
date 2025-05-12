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



def evaluate_two_teams_by_game_id(game_id, df_matches, df_players, df_fifa, df_lineups):

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

def get_team_points(df: pd.DataFrame, game_id) -> tuple[float, float]:
    """
    Pre dané game_id vráti (home_points, away_points), kde každá hodnota je
    normalizované body získané domacim resp. hosťujúcim tímom vo všetkých zápasoch
    pred daným zápasom v rovnakej súťaži a sezóne, v rozmedzí 0 až 1.

    Normalizácia: actual_points / (3 * number_of_games_played).
    Body: výhra=3, remíza=1, prehra=0.
    Ak tím nemá žiadne predchádzajúce zápasy, vráti (np.nan, np.nan).
    """
    df2 = df.copy()
    for col in ['home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    df2['date'] = pd.to_datetime(df2['date'], dayfirst=True, errors='coerce')

    m0 = df2[df2['game_id'] == game_id]
    if m0.empty:
        return np.nan, np.nan
    m0 = m0.iloc[0]
    date0 = m0['date']
    comp, season = m0['competition_id'], m0['season']
    home_id = m0['home_club_id']
    away_id = m0['away_club_id']

    past = df2[(df2['competition_id'] == comp) &
               (df2['season'] == season) &
               (df2['date'] < date0)]

    def norm(team_id: float) -> float:
        sub = past[(past['home_club_id'] == team_id) | (past['away_club_id'] == team_id)]
        if sub.empty:
            return np.nan
        pts = 0
        games = 0
        for _, r in sub.iterrows():
            gf = r['home_club_goals'] if r['home_club_id'] == team_id else r['away_club_goals']
            ga = r['away_club_goals'] if r['home_club_id'] == team_id else r['home_club_goals']
            if pd.isna(gf) or pd.isna(ga):
                continue
            games += 1
            if gf > ga:
                pts += 3
            elif gf == ga:
                pts += 1
        return pts / (3 * games) if games > 0 else np.nan

    return norm(home_id), norm(away_id)


def get_form_points(df: pd.DataFrame, game_id: str, form_n: int = 10) -> tuple[float, float]:
    """
    Pre dané game_id vráti (home_form_norm, away_form_norm) z posledných form_n zápasov.
    Normalizácia: actual_points / (3 * number_of_games_played).
    Ak tím nemá žiadne predchádzajúce zápasy, vráti np.nan.
    """
    df2 = df.copy()
    for col in ['home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    df2['date'] = pd.to_datetime(df2['date'], dayfirst=True, errors='coerce')

    m0 = df2[df2['game_id'] == game_id]
    if m0.empty:
        return np.nan, np.nan
    m0 = m0.iloc[0]
    date0 = m0['date']
    comp, season = m0['competition_id'], m0['season']
    home_id, away_id = m0['home_club_id'], m0['away_club_id']

    past = df2[(df2['competition_id'] == comp) &
               (df2['season'] == season) &
               (df2['date'] < date0)]

    home_sub = past[(past['home_club_id'] == home_id) | (past['away_club_id'] == home_id)]
    away_sub = past[(past['home_club_id'] == away_id) | (past['away_club_id'] == away_id)]
    home_sub = home_sub.sort_values('date', ascending=False).head(form_n)
    away_sub = away_sub.sort_values('date', ascending=False).head(form_n)

    def norm(sub, team_id):
        if sub.empty:
            return np.nan
        pts = 0
        games = 0
        for _, r in sub.iterrows():
            gf = r['home_club_goals'] if r['home_club_id'] == team_id else r['away_club_goals']
            ga = r['away_club_goals'] if r['home_club_id'] == team_id else r['home_club_goals']
            if pd.isna(gf) or pd.isna(ga):
                continue
            games += 1
            if gf > ga:
                pts += 3
            elif gf == ga:
                pts += 1
        return pts / (3 * games) if games > 0 else np.nan

    return norm(home_sub, home_id), norm(away_sub, away_id)


def get_result_rate(df: pd.DataFrame, game_id: str) -> tuple[float, float, float, float]:
    """
    Pre dané game_id vráti win/draw rate pre domáce a vonkajšie zápasy.
    Ak tím nemá žiadne predchádzajúce zápasy v danej kategórii, vráti np.nan pre príslušné miery.
    """
    df2 = df.copy()
    for col in ['home_club_goals', 'away_club_goals', 'home_club_id', 'away_club_id']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    df2['date'] = pd.to_datetime(df2['date'], dayfirst=True, errors='coerce')

    m0 = df2[df2['game_id'] == game_id]
    if m0.empty:
        return np.nan, np.nan, np.nan, np.nan
    m0 = m0.iloc[0]
    comp, season, date0 = m0['competition_id'], m0['season'], m0['date']
    home_id, away_id = m0['home_club_id'], m0['away_club_id']

    past = df2[(df2['competition_id'] == comp) &
               (df2['season'] == season) &
               (df2['date'] < date0)]

    home_p = past[past['home_club_id'] == home_id].dropna(subset=['home_club_goals','away_club_goals'])
    away_p = past[past['away_club_id'] == away_id].dropna(subset=['home_club_goals','away_club_goals'])

    def rate(sub, gf_col, ga_col):
        total = len(sub)
        if total == 0:
            return np.nan, np.nan
        wins = (sub[gf_col] > sub[ga_col]).sum()
        draws = (sub[gf_col] == sub[ga_col]).sum()
        return wins/total, draws/total

    h_wr, h_dr = rate(home_p, 'home_club_goals', 'away_club_goals')
    a_wr, a_dr = rate(away_p, 'away_club_goals', 'home_club_goals')
    return h_wr, h_dr, a_wr, a_dr


def get_mutual_statistic(df: pd.DataFrame, game_id: str) -> tuple[float, float]:
    """
    Head-to-head win rate pre oba tímy.
    Vráti (h2h_home_win_rate, h2h_away_win_rate). Ak vzájomné zápasy neexistujú, vráti (np.nan, np.nan).
    """
    df2 = df.copy()
    for col in ['home_club_goals', 'away_club_goals', 'home_club_id', 'away_club_id']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    df2['date'] = pd.to_datetime(df2['date'], dayfirst=True, errors='coerce')

    m0 = df2[df2['game_id'] == game_id]
    if m0.empty:
        return np.nan, np.nan
    m0 = m0.iloc[0]
    comp, season, date0 = m0['competition_id'], m0['season'], m0['date']
    home_id, away_id = m0['home_club_id'], m0['away_club_id']

    past = df2[(df2['competition_id'] == comp) &
               (df2['season'] == season) &
               (df2['date'] < date0)]
    mutual = past[((past['home_club_id'] == home_id) & (past['away_club_id'] == away_id)) |
                  ((past['home_club_id'] == away_id) & (past['away_club_id'] == home_id))]
    mutual = mutual.dropna(subset=['home_club_goals','away_club_goals'])

    total = len(mutual)
    if total == 0:
        return np.nan, np.nan
    home_wins = ((mutual['home_club_id'] == home_id) & (mutual['home_club_goals'] > mutual['away_club_goals'])).sum() + \
                ((mutual['away_club_id'] == home_id) & (mutual['away_club_goals'] > mutual['home_club_goals'])).sum()
    away_wins = ((mutual['home_club_id'] == away_id) & (mutual['home_club_goals'] > mutual['away_club_goals'])).sum() + \
                ((mutual['away_club_id'] == away_id) & (mutual['away_club_goals'] > mutual['home_club_goals'])).sum()
    return home_wins / total, away_wins / total

def get_days_rest(df: pd.DataFrame, game_id: str) -> tuple[float, float]:
    """
    Pre dané game_id vráti (home_days_rest, away_days_rest), kde každá hodnota je počet dní
    od predchádzajúceho zápasu domáceho resp. hosťujúceho tímu v rovnakej súťaži a sezóne.
    Ak tím ešte nemal predchádzajúci zápas, vráti np.nan.

    Parametre:
    - df: DataFrame obsahujúci stĺpce 'game_id', 'competition_id', 'season', 'date',
           'home_club_id', 'away_club_id'.
    - game_id: identifikátor zápasu, pre ktorý počítame rest days.
    """
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date'], dayfirst=True, errors='coerce')

    match = df2[df2['game_id'] == game_id]
    if match.empty:
        raise ValueError(f"Game ID {game_id} not found in DataFrame.")
    m0 = match.iloc[0]
    date0 = m0['date']
    comp = m0['competition_id']
    season = m0['season']
    home_id = m0['home_club_id']
    away_id = m0['away_club_id']

    past = df2[(df2['competition_id'] == comp) &
               (df2['season'] == season) &
               (df2['date'] < date0)]

    def days_since_last(team_id):
        team_games = past[(past['home_club_id'] == team_id) | (past['away_club_id'] == team_id)]
        if team_games.empty:
            return np.nan
        last_date = team_games['date'].max()
        return (date0 - last_date).days

    home_rest = days_since_last(home_id)
    away_rest = days_since_last(away_id)
    return home_rest, away_rest

def get_average_goals_scored(df: pd.DataFrame, game_id: str, form_n: int = 5) -> tuple[float, float]:
    """
    Pre dané game_id vráti (home_avg_scored, away_avg_scored), kde každá hodnota je
    priemerný počet strelených gólov na zápas z posledných form_n zápasov v rovnakej súťaži a sezóne.
    Ak tím nemá žiadne predchádzajúce zápasy, vráti (np.nan, np.nan).
    """
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date'], dayfirst=True, errors='coerce')

    m0 = df2[df2['game_id'] == game_id]
    if m0.empty:
        return np.nan, np.nan
    m0 = m0.iloc[0]
    date0 = m0['date']
    comp, season = m0['competition_id'], m0['season']
    home_id, away_id = m0['home_club_id'], m0['away_club_id']

    past = df2[(df2['competition_id'] == comp) &
               (df2['season'] == season) &
               (df2['date'] < date0)]

    def avg_scored(team_id):
        sub = past[(past['home_club_id'] == team_id) | (past['away_club_id'] == team_id)]
        sub = sub.sort_values('date', ascending=False).head(form_n)
        if sub.empty:
            return np.nan
        scored = sub.apply(lambda r: (r['home_club_goals'] if r['home_club_id'] == team_id else r['away_club_goals']), axis=1)
        return scored.mean()

    home_avg = avg_scored(home_id)
    away_avg = avg_scored(away_id)
    return home_avg, away_avg


def get_average_goals_conceded(df: pd.DataFrame, game_id: str, form_n: int = 5) -> tuple[float, float]:
    """
    Pre dané game_id vráti (home_avg_conceded, away_avg_conceded), kde každá hodnota je
    priemerný počet inkasovaných gólov na zápas z posledných form_n zápasov v rovnakej súťaži a sezóne.
    Ak tím nemá žiadne predchádzajúce zápasy, vráti (np.nan, np.nan).
    """
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date'], dayfirst=True, errors='coerce')

    m0 = df2[df2['game_id'] == game_id]
    if m0.empty:
        return np.nan, np.nan
    m0 = m0.iloc[0]
    date0 = m0['date']
    comp, season = m0['competition_id'], m0['season']
    home_id, away_id = m0['home_club_id'], m0['away_club_id']

    past = df2[(df2['competition_id'] == comp) &
               (df2['season'] == season) &
               (df2['date'] < date0)]

    def avg_conceded(team_id):
        sub = past[(past['home_club_id'] == team_id) | (past['away_club_id'] == team_id)]
        sub = sub.sort_values('date', ascending=False).head(form_n)
        if sub.empty:
            return np.nan
        conceded = sub.apply(lambda r: (r['away_club_goals'] if r['home_club_id'] == team_id else r['home_club_goals']), axis=1)
        return conceded.mean()

    home_avg_con = avg_conceded(home_id)
    away_avg_con = avg_conceded(away_id)
    return home_avg_con, away_avg_con

def get_mutual_goal_difference(df: pd.DataFrame, game_id: str) -> tuple[float, float]:
    """
    Pre dané game_id vráti (home_goal_diff, away_goal_diff), kde hodnota je
    (goals_for - goals_against) v predchádzajúcich vzájomných zápasoch
    v rovnakej súťaži a sezóne. Ak neexistujú mutual zápasy, vráti (np.nan, np.nan).
    """
    df2 = df.copy()
    for col in ['home_club_goals', 'away_club_goals', 'home_club_id', 'away_club_id']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    df2['date'] = pd.to_datetime(df2['date'], dayfirst=True, errors='coerce')
    m0 = df2[df2['game_id'] == game_id]
    if m0.empty:
        return np.nan, np.nan
    m0 = m0.iloc[0]
    comp, season, date0 = m0['competition_id'], m0['season'], m0['date']
    home_id, away_id = m0['home_club_id'], m0['away_club_id']
    past = df2[(df2['competition_id'] == comp) &
               (df2['season'] == season) &
               (df2['date'] < date0)]
    mutual = past[((past['home_club_id'] == home_id) & (past['away_club_id'] == away_id)) |
                  ((past['home_club_id'] == away_id) & (past['away_club_id'] == home_id))]
    if mutual.empty:
        return np.nan, np.nan
    home_for = ((mutual['home_club_id'] == home_id) * mutual['home_club_goals'] +
                (mutual['away_club_id'] == home_id) * mutual['away_club_goals']).sum()
    home_against = ((mutual['home_club_id'] == home_id) * mutual['away_club_goals'] +
                    (mutual['away_club_id'] == home_id) * mutual['home_club_goals']).sum()
    away_for = ((mutual['home_club_id'] == away_id) * mutual['home_club_goals'] +
                (mutual['away_club_id'] == away_id) * mutual['away_club_goals']).sum()
    away_against = ((mutual['home_club_id'] == away_id) * mutual['away_club_goals'] +
                    (mutual['away_club_id'] == away_id) * mutual['home_club_goals']).sum()
    home_diff = home_for - home_against
    away_diff = away_for - away_against
    return home_diff, away_diff

def get_current_league_points(df: pd.DataFrame, game_id: str) -> tuple[int, int]:
    """
    Pre dané game_id vráti (home_raw_points, away_raw_points), kde každá hodnota je
    súčet bodov (výhra=3, remíza=1, prehra=0) získaných domacim resp. hosťujúcim tímom
    vo všetkých predchádzajúcich zápasoch v rovnakej súťaži a sezóne.
    Ak tím nemá žiadne predchádzajúce zápasy, vráti (0, 0).
    """
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date'], dayfirst=True, errors='coerce')

    m0 = df2[df2['game_id'] == game_id]
    if m0.empty:
        return 0, 0
    m0 = m0.iloc[0]
    date0 = m0['date']
    comp, season = m0['competition_id'], m0['season']
    home_id, away_id = m0['home_club_id'], m0['away_club_id']

    past = df2[(df2['competition_id'] == comp) &
               (df2['season'] == season) &
               (df2['date'] < date0)]
    def raw_points(team_id):
        sub = past[(past['home_club_id'] == team_id) | (past['away_club_id'] == team_id)]
        if sub.empty:
            return 0
        pts = 0
        for _, r in sub.iterrows():
            gf = r['home_club_goals'] if r['home_club_id'] == team_id else r['away_club_goals']
            ga = r['away_club_goals'] if r['home_club_id'] == team_id else r['home_club_goals']
            if pd.isna(gf) or pd.isna(ga):
                continue
            if gf > ga:
                pts += 3
            elif gf == ga:
                pts += 1
        return pts

    return raw_points(home_id), raw_points(away_id)


def get_goal_difference_momentum(df: pd.DataFrame, game_id: str, form_n: int = 5) -> tuple[float, float]:
    """
    Pre dané game_id vráti (home_goal_diff_mom, away_goal_diff_mom), kde každá hodnota je
    priemerný rozdiel gólov (goals_for - goals_against) za posledných form_n zápasov
    v rovnakej súťaži a sezóne. Ak tím nemá žiadne predchádzajúce zápasy, vráti np.nan.
    """
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date'], dayfirst=True, errors='coerce')

    m0 = df2[df2['game_id'] == game_id]
    if m0.empty:
        return np.nan, np.nan
    m0 = m0.iloc[0]
    date0 = m0['date']
    comp, season = m0['competition_id'], m0['season']
    home_id, away_id = m0['home_club_id'], m0['away_club_id']

    past = df2[(df2['competition_id'] == comp) &
               (df2['season'] == season) &
               (df2['date'] < date0)]

    def momentum(team_id):
        sub = past[(past['home_club_id'] == team_id) | (past['away_club_id'] == team_id)]
        sub = sub.sort_values('date', ascending=False).head(form_n)
        if sub.empty:
            return np.nan
        diffs = sub.apply(lambda r: (r['home_club_goals'] - r['away_club_goals']) if r['home_club_id'] == team_id \
                                         else (r['away_club_goals'] - r['home_club_goals']), axis=1)
        return diffs.sum() / len(diffs)

    return momentum(home_id), momentum(away_id)

def get_clean_sheet_rate(df: pd.DataFrame, game_id: str, form_n: int = 5) -> tuple[float, float]:
    """
    Pre dané game_id vráti (home_clean_sheet_rate, away_clean_sheet_rate), kde každá hodnota je
    pomer zápasov v posledných form_n zápasoch, v ktorých tím inkasoval 0 gólov,
    v rovnakej súťaži a sezóne. Ak tím nemá žiadne predchádzajúce zápasy, vráti np.nan.
    """
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date'], dayfirst=True, errors='coerce')

    m0 = df2[df2['game_id'] == game_id]
    if m0.empty:
        return np.nan, np.nan
    m0 = m0.iloc[0]
    date0 = m0['date']
    comp, season = m0['competition_id'], m0['season']
    home_id, away_id = m0['home_club_id'], m0['away_club_id']

    past = df2[(df2['competition_id'] == comp) &
               (df2['season'] == season) &
               (df2['date'] < date0)]

    def clean_rate(team_id):
        sub = past[(past['home_club_id'] == team_id) | (past['away_club_id'] == team_id)]
        sub = sub.sort_values('date', ascending=False).head(form_n)
        if sub.empty:
            return np.nan
        conceded = sub.apply(lambda r: r['away_club_goals'] if r['home_club_id'] == team_id else r['home_club_goals'], axis=1)
        clean = (conceded == 0).sum()
        return clean / len(conceded)

    return clean_rate(home_id), clean_rate(away_id)