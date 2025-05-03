import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
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
df_matches = pd.read_csv("https://drive.google.com/uc?export=download&id=1VLG5_Nf8yUeNt30ZI10S3Ay83rTZTj2h")
df_players = pd.read_csv("https://drive.google.com/uc?export=download&id=1wQhC7QTu1hnkgy32JAOOxipYXGWW6bvF")
df_teams   = pd.read_csv("https://drive.google.com/uc?export=download&id=1gRBU_77__Oiz3l16xTjFbPh6DkKMnFcY")
df_competitions = pd.read_csv("https://drive.google.com/uc?export=download&id=1X9gnXZE_kzfa8xvqVe3qnYbeberWVsGG")

config = utils.loadConfig()
if (config["is_training"]):
    training.train_model(config)

if (config["is_gui"]):
    gui.loadGUI(df_teams, df_competitions, df_players, utils.loadModel())
