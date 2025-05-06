import pandas as pd
from Features.player_features import evaluate_two_player_dfs
from Features.team_features import evaluate_two_teams_by_game_id, get_form_points, get_form_points_by_team, get_result_rate, get_result_rate_by_team

def get_dataframe_match(
    df_matches: pd.DataFrame,
    df_fifa: pd.DataFrame,
    df_lineup_home: pd.DataFrame,
    df_lineup_away: pd.DataFrame,
    comp: int,
    season: int,
    date: pd.Timestamp,
    home_id: int,
    away_id: int,
    form_n: int
) -> pd.DataFrame:
    """
    Create a single-row feature dataframe from match details and team lineups.
    """
    frame = evaluate_two_player_dfs(df_lineup_home, df_lineup_away, df_fifa)
    home_form, away_form = get_form_points_by_team(comp, season, date, df_matches, home_id, away_id, form_n)
    home_win_rate, home_draw_rate, away_win_rate, away_draw_rate = get_result_rate_by_team(df_matches, comp, season, date, home_id, away_id)

    frame['home_form'] = [home_form]
    frame['away_form'] = [away_form]
    frame['home_win_rate'] = [home_win_rate]
    frame['home_draw_rate'] = [home_draw_rate]
    frame['away_win_rate'] = [away_win_rate]
    frame['away_draw_rate'] = [away_draw_rate]
    return frame

def get_dataframe_game_id(
    game_id: str,
    df_matches: pd.DataFrame,
    df_players: pd.DataFrame,
    df_fifa: pd.DataFrame,
    df_lineups: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a feature dataframe using just the game ID and main dataframes.
    """
    frame = evaluate_two_teams_by_game_id(game_id, df_matches, df_players, df_fifa, df_lineups)
    """  home_form, away_form = get_form_points(df_matches, game_id)
    home_win_rate, home_draw_rate, away_win_rate, away_draw_rate = get_result_rate(df_matches, game_id)

    frame['home_form'] = [home_form]
    frame['away_form'] = [away_form]
    frame['home_win_rate'] = [home_win_rate]
    frame['home_draw_rate'] = [home_draw_rate]
    frame['away_win_rate'] = [away_win_rate]
    frame['away_draw_rate'] = [away_draw_rate] """
    return frame