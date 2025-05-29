import pandas as pd

import  Features.match_features
import  Features.team_features 

def createDataset(df_fifa: pd.DataFrame,
                df_lineups: pd.DataFrame,
                df_matches: pd.DataFrame,
                df_players: pd.DataFrame) -> pd.DataFrame:
    frames = []

    df_matches["date"] = pd.to_datetime(df_matches["date"], dayfirst=True, errors='coerce')
    i = 0
    for _, game in df_matches.iterrows():
        frame = Features.match_features.get_dataframe_game_id(game["game_id"], df_matches, df_players, df_fifa, df_lineups)
        first_24_cols = frame.columns[:24]
            
        has_valid_data = frame[first_24_cols].map(lambda x: pd.notna(x) and x != 0.0).any(axis=1).any()

        if frame is not None and not frame.empty and has_valid_data:
            game_id = game["game_id"]
            frame['home_goals'] = game['home_club_goals']
            frame['away_goals'] = game['away_club_goals']
            frame['competition_id'] = game['competition_id']
            home_points, away_points = Features.team_features.get_team_points(df_matches, game_id)
            h_wr, h_dr, a_wr, a_dr = Features.team_features.get_result_rate(df_matches, game_id)
            h2h_home, h2h_away = Features.team_features.get_mutual_statistic(df_matches, game_id)
            home_rest_days, away_rest_days = Features.team_features.get_days_rest(df_matches, game_id)
            home_scored, away_scored = Features.team_features.get_average_goals_scored(df_matches, game_id)
            home_conceded, away_conceded = Features.team_features.get_average_goals_conceded(df_matches, game_id)
            home_diff, away_diff = Features.team_features.get_mutual_goal_difference(df_matches, game_id)
            home_raw_points, away_raw_points = Features.team_features.get_current_league_points(df_matches, game_id)
            home_momentum, away_momentum = Features.team_features.get_goal_difference_momentum(df_matches, game_id)
            home_clean_sheet, away_clean_sheet = Features.team_features.get_clean_sheet_rate(df_matches, game_id)

            feature_row = [
                home_points, away_points,
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

            frames.append(combined)
            print(i, " - ", game["game_id"])
            i += 1



    datset = pd.concat(frames, ignore_index=True)
    print(datset.columns.to_list())
    return datset