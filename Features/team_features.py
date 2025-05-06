import pandas as pd
from Features.player_features import calculate_position_stats, evaluate_two_player_dfs, get_players_evaluation_df

def evaluate_two_teams_by_game_id(
    game_id: str,
    df_matches: pd.DataFrame,
    df_players: pd.DataFrame,
    df_fifa: pd.DataFrame,
    df_lineups: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluate the match-up between two teams based on a given game_id.
    Returns a single-row dataframe of engineered features from both teams' players.
    """
    df_matches_filtered = df_matches[df_matches['game_id'] == game_id]
    home_club_id = df_matches_filtered['home_club_id'].iloc[0]  
    away_club_id = df_matches_filtered['away_club_id'].iloc[0] 

    player_ids_home = df_lineups.loc[
        (df_lineups['game_id'] == game_id) & (df_lineups['club_id'] == home_club_id),
        'player_id']

    player_ids_away = df_lineups.loc[
        (df_lineups['game_id'] == game_id) & (df_lineups['club_id'] == away_club_id),
        'player_id']

    df_players_1 = df_players[df_players['player_id'].isin(player_ids_home)].copy()
    df_players_2 = df_players[df_players['player_id'].isin(player_ids_away)].copy()

    df_players_1['game_team_id'] = home_club_id 
    df_players_2['game_team_id'] = away_club_id

    return evaluate_two_player_dfs(df_players_1, df_players_2, df_fifa)

def get_team_points(df: pd.DataFrame, game_id: str) -> tuple[float, float]:
    """
    Return normalized total points for home and away teams before the given game.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    match = df.loc[df['game_id'] == game_id].iloc[0]
    comp, season, date0, home, away = match['competition_id'], match['season'], match['date'], match['home_club_id'], match['away_club_id']

    past = df[(df['competition_id'] == comp) & (df['season'] == season) & (df['date'] < date0)]

    def normalized_points_for(team_id: int) -> float:
        sub = past[(past['home_club_id'] == team_id) | (past['away_club_id'] == team_id)]
        pts = sum(
            3 if (r['home_club_id'] == team_id and r['home_club_goals'] > r['away_club_goals']) or
                 (r['away_club_id'] == team_id and r['away_club_goals'] > r['home_club_goals'])
            else 1 if (r['home_club_goals'] == r['away_club_goals']) else 0
            for _, r in sub.iterrows()
        )
        max_pts = 3 * len(sub)
        return pts / max_pts if max_pts > 0 else 0.0

    return normalized_points_for(home), normalized_points_for(away)


def get_form_points(
    df: pd.DataFrame,
    game_id: str,
    form_n: int = 10
) -> tuple[float, float]:
    """
    Return normalized points in the last 'form_n' matches for both teams before the game.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    match = df.loc[df['game_id'] == game_id].iloc[0]
    return get_form_points_by_team(
        comp=match['competition_id'],
        season=match['season'],
        date=match['date'],
        df_matches=df,
        home_id=match['home_club_id'],
        away_id=match['away_club_id'],
        form_n=form_n
    )

def get_form_points_by_team(
    comp: int,
    season: int,
    date: pd.Timestamp,
    df_matches: pd.DataFrame,
    home_id: int,
    away_id: int,
    form_n: int
) -> tuple[float, float]:
    """
    Compute normalized form for both teams in the 'form_n' matches before the given date.
    """
    past = df_matches[(df_matches['competition_id'] == comp) & (df_matches['season'] == season) & (df_matches['date'] < date)]

    def get_normalized_form(team_id: int) -> float:
        sub = past[(past['home_club_id'] == team_id) | (past['away_club_id'] == team_id)].sort_values('date', ascending=False).head(form_n)
        pts = sum(
            3 if (r['home_club_id'] == team_id and r['home_club_goals'] > r['away_club_goals']) or
                 (r['away_club_id'] == team_id and r['away_club_goals'] > r['home_club_goals'])
            else 1 if (r['home_club_goals'] == r['away_club_goals']) else 0
            for _, r in sub.iterrows()
        )
        return pts / (3 * len(sub)) if len(sub) > 0 else 0.0

    return get_normalized_form(home_id), get_normalized_form(away_id)

def get_result_rate(
    df: pd.DataFrame,
    game_id: str
) -> tuple[float, float, float, float]:
    """
    Return historical win/draw rates for both home and away teams.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    match = df.loc[df['game_id'] == game_id].iloc[0]
    return get_result_rate_by_team(
        df,
        comp=match['competition_id'],
        season=match['season'],
        date=match['date'],
        home_id=match['home_club_id'],
        away_id=match['away_club_id']
    )

def get_result_rate_by_team(
    df: pd.DataFrame,
    comp: int,
    season: int,
    date: pd.Timestamp,
    home_id: int,
    away_id: int
) -> tuple[float, float, float, float]:
    """
    Return win/draw rates for home and away teams based on past matches.
    """
    past = df[(df['competition_id'] == comp) & (df['season'] == season) & (df['date'] < date)]

    def rate(sub: pd.DataFrame, goals_col: str, opp_goals_col: str) -> tuple[float, float]:
        total = len(sub)
        wins = (sub[goals_col] > sub[opp_goals_col]).sum()
        draws = (sub[goals_col] == sub[opp_goals_col]).sum()
        return wins / total if total else 0.0, draws / total if total else 0.0

    home_wins, home_draws = rate(past[past['home_club_id'] == home_id], 'home_club_goals', 'away_club_goals')
    away_wins, away_draws = rate(past[past['away_club_id'] == away_id], 'away_club_goals', 'home_club_goals')
    return home_wins, home_draws, away_wins, away_draws



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