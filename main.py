import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import graphs
import gui
import Model.training as training
import utils

#datasets
df_fifa = pd.read_csv("Data/fifa_players.csv")
df_lineups = pd.read_csv('Data/game_lineups.csv')
#transfermarkt
df_matches = pd.read_csv("Data/games.csv")
df_players = pd.read_csv("Data/players.csv")
df_teams   = pd.read_csv("Data/clubs.csv")
df_competitions = pd.read_csv("Data/competitions.csv")

config = utils.loadConfig()
if (config["show_graphs"]):
    graphs.featureCorrelation(df_fifa)
    graphs.ratingToAge(df_fifa)
    graphs.avgRatingByNationality(df_fifa)

if (config["is_training"]):
    training.trainModel(config,df_fifa, df_lineups, df_matches, df_players, df_teams, df_competitions)

if (config["is_gui"]):
    gui.loadGUI(df_teams, df_competitions, df_players, None)

utils.save_config(config)
