import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import graphs
import gui
from model import MatchPredictorFCNN, evaluate_model, train_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import fbrefdata as fd
import training
import utils

df_fifa = pd.read_csv("https://drive.google.com/uc?export=download&id=1Fg06rxyQ-ImEf_CKfOoxZEqAIFIsjvZN")
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

if (config["is_training"]):
    training.train_model(config)

if (config["is_gui"]):
    gui.loadGUI(df_teams, df_competitions, df_players, utils.loadModel())
