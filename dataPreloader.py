import pandas as pd

from Features.match_features import get_dataframe_game_id

def createDataset(df_fifa: pd.DataFrame,
                df_lineups: pd.DataFrame,
                df_matches: pd.DataFrame,
                df_players: pd.DataFrame) -> pd.DataFrame:
    frames = []

    # Filter only Premier League matches (assuming "GB1" is the ID for that)
    df_matches["date"] = pd.to_datetime(df_matches["date"], dayfirst=True, errors='coerce')
    #df_matches = df_matches[
     #   (df_matches["competition_id"] == "GB1") & 
     #   (df_matches["date"] > "2015-10-28")
    #]

    for _, game in df_matches.iterrows():
        frame = get_dataframe_game_id(game["game_id"], df_matches, df_players, df_fifa, df_lineups)
        if frame is not None and not frame.empty:
            frame['home_goals'] = game['home_club_goals']
            frame['away_goals'] = game['away_club_goals']

            frames.append(frame)


    datset = pd.concat(frames, ignore_index=True)
    return datset