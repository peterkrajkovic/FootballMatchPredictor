import tkinter as tk
from tkinter import ttk, messagebox
from pandas import DataFrame
import torch
import numpy as np
from model import MatchPredictorFCNN

def loadGUI(df_teams : DataFrame, df_competitions : DataFrame, df_players : DataFrame, model : MatchPredictorFCNN):
    # Load team, league, and player info
    team_names = df_teams['name'].unique().tolist()
    league_names = df_competitions['name'].unique().tolist()
    player_names = df_players['name'].unique().tolist()
    
    def predict_match():
        team1 = team1_combo.get()
        team2 = team2_combo.get()
        league = league_combo.get()
        players_team1 = [players_listbox_team1.get(i) for i in players_listbox_team1.curselection()]
        players_team2 = [players_listbox_team2.get(i) for i in players_listbox_team2.curselection()]

        if not team1 or not team2 or not league or not players_team1 or not players_team2:
            messagebox.showerror("Input Error", "Please make all selections.")
            return

        # --- Placeholder: Create feature vector ---
        features = torch.randn(1, 10)

        # Predict
        with torch.no_grad():
            output = model(features)
            prediction = torch.argmax(output, dim=1).item()

        result_text = ["Home Win", "Draw", "Away Win"][prediction]
        result_label.config(text=f"Predicted Result: {result_text}")

    root = tk.Tk()
    root.title("Football Match Predictor")

    ttk.Label(root, text="Select League:").grid(row=0, column=0, padx=5, pady=5)
    league_combo = ttk.Combobox(root, values=league_names)
    league_combo.grid(row=0, column=1)

    ttk.Label(root, text="Team 1 (Home):").grid(row=1, column=0)
    team1_combo = ttk.Combobox(root, values=team_names)
    team1_combo.grid(row=1, column=1)

    ttk.Label(root, text="Team 2 (Away):").grid(row=2, column=0)
    team2_combo = ttk.Combobox(root, values=team_names)
    team2_combo.grid(row=2, column=1)

    ttk.Label(root, text="Players Team 1:").grid(row=3, column=0)
    players_listbox_team1 = tk.Listbox(root, selectmode='multiple', height=10)
    players_listbox_team1.grid(row=4, column=0, padx=5, pady=5)
    for player in player_names:
        players_listbox_team1.insert(tk.END, player)

    ttk.Label(root, text="Players Team 2:").grid(row=3, column=1)
    players_listbox_team2 = tk.Listbox(root, selectmode='multiple', height=10)
    players_listbox_team2.grid(row=4, column=1, padx=5, pady=5)
    for player in player_names:
        players_listbox_team2.insert(tk.END, player)

    predict_button = ttk.Button(root, text="Predict Match", command=predict_match)
    predict_button.grid(row=5, column=0, columnspan=2, pady=10)

    result_label = ttk.Label(root, text="Predicted Result: ")
    result_label.grid(row=6, column=0, columnspan=2)

    root.mainloop()

