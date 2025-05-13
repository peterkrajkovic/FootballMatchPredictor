import pandas as pd
from Features.player_features import calculate_position_stats, evaluate_two_player_dfs, get_players_evaluation_df
import numpy as np

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

