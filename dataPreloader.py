import pandas as pd

from Features.match_features import get_dataframe_game_id

def createDataset(df_fifa: pd.DataFrame,
                df_lineups: pd.DataFrame,
                df_matches: pd.DataFrame,
                df_players: pd.DataFrame) -> pd.DataFrame:
    frames = []

    df_matches["date"] = pd.to_datetime(df_matches["date"], dayfirst=True, errors='coerce')
    i = 0
    for _, game in df_matches.iterrows():
        frame = get_dataframe_game_id(game["game_id"], df_matches, df_players, df_fifa, df_lineups)
        if frame is not None and not frame.empty:
            frame['home_goals'] = game['home_club_goals']
            frame['away_goals'] = game['away_club_goals']

            frames.append(frame)
            i += 1


            print(i, " - ", game["game_id"])



    datset = pd.concat(frames, ignore_index=True)
    return datset