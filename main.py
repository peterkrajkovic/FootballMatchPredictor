import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import graphs
import gui
import Model.training as training
import utils
import dataPreloader

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

if (config["preload_data"]):
    dataset = dataPreloader.createDataset(df_fifa, df_lineups, df_matches, df_players)
    dataset.to_csv("Data/dataset_ultra_large.csv", index=False)
    print("Creating and saving dataset.csv ")



if (config["is_training"]):

    #'S1', 'GB1', 'FR1', 'ES1', 'NL1', 'IT1'

   # dataset = pd.read_csv("Data/dataset_with_competition_id.csv")
    #filtered_dataset = dataset[dataset['competition_id'].isin([ 'GB1'])]

   # first_24_cols = filtered_dataset.columns[:24]
   # mask_all_zero_or_nan = filtered_dataset[first_24_cols].applymap(lambda x: pd.isna(x) or x == 0.0).all(axis=1)
    #features_cleaned = filtered_dataset[~mask_all_zero_or_nan]


    #training.prepareModel(config, features_cleaned)

     

    with open('Data/labels_all_cID.txt', 'r') as f:
        labels = [float(line.strip()) for line in f]
    features = pd.read_csv("Data/features_all_cID.csv")

    features = features.drop('competition_id', axis=1)

    training.trainModel(config, features, labels)


if (config["is_gui"]):
    gui.loadGUI(df_teams, df_competitions, df_players, None)

utils.save_config(config)
