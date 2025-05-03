import tkinter as tk
from tkinter import ttk, messagebox
from pandas import DataFrame
import torch
from model import MatchPredictorFCNN

def loadGUI(df_teams: DataFrame, df_competitions: DataFrame, df_players: DataFrame, model: MatchPredictorFCNN):
    root = tk.Tk()
    root.title("Football Match Predictor")
    root.geometry("1200x900")

    team_names = df_teams['name'].unique().tolist()
    league_names = df_competitions['name'].unique().tolist()

    selected_counts = {
        'Team1': tk.StringVar(value='0/15'),
        'Team2': tk.StringVar(value='0/15')
    }

    def update_team_selection(event=None):
        league = league_combo.get()
        if not league:
            return
        competition_id = df_competitions[df_competitions['name'] == league]['competition_id'].values[0]
        teams = df_teams[df_teams['domestic_competition_id'] == competition_id]
        filtered_team_names = teams['name'].tolist()
        team1_combo['values'] = filtered_team_names
        team2_combo['values'] = filtered_team_names

    def update_player_lists(event=None):
        combo = event.widget
        team_name = combo.get()

        if combo == team1_combo:
            target_listboxes = players_team1_listboxes
            team_key = 'Team1'
        else:
            target_listboxes = players_team2_listboxes
            team_key = 'Team2'

        # Clear all listboxes
        for lb in target_listboxes.values():
            lb.delete(0, tk.END)

        filtered = df_players[df_players['current_club_name'] == team_name]

        already_added = set()
        for _, player in filtered.iterrows():
            name = player['name']
            pos = player['position']
            already_added.add(name)
            if pos == 'Goalkeeper':
                target_listboxes['Goalkeepers'].insert(tk.END, name)
            elif pos in ['Centre-Back', 'Full-Back', 'Defender']:
                target_listboxes['Defenders'].insert(tk.END, name)
            elif pos in ['Central Midfield', 'Defensive Midfield', 'Attacking Midfield', 'Wide Midfield', 'Midfield']:
                target_listboxes['Midfielders'].insert(tk.END, name)
            elif pos in ['Centre-Forward', 'Centre-Attack', 'Winger', 'Attack']:
                target_listboxes['Forwards'].insert(tk.END, name)

        # Populate substitutes with unselected players
        for name in already_added:
            target_listboxes['Substitutes'].insert(tk.END, name)

    def get_selected_players(listboxes):
        selected = set()
        for key in ['Goalkeepers', 'Defenders', 'Midfielders', 'Forwards']:
            lb = listboxes[key]
            selected.update(lb.get(i) for i in lb.curselection())
        return selected

    def update_counts(*args):
        count1 = len(get_selected_players(players_team1_listboxes))
        count2 = len(get_selected_players(players_team2_listboxes))
        selected_counts['Team1'].set(f'{count1}/15')
        selected_counts['Team2'].set(f'{count2}/15')

    def predict_match():
        team1 = team1_combo.get()
        team2 = team2_combo.get()
        league = league_combo.get()
        players_team1 = get_selected_players(players_team1_listboxes)
        players_team2 = get_selected_players(players_team2_listboxes)

        if not team1 or not team2 or not league:
            messagebox.showerror("Input Error", "Please make all selections.")
            return
        if len(players_team1) > 15 or len(players_team2) > 15:
            messagebox.showerror("Input Error", "Each team can have a maximum of 15 players.")
            return

        features = torch.randn(1, 10)  # Placeholder
        with torch.no_grad():
            output = model(features)
            prediction = torch.argmax(output, dim=1).item()

        result_text = ["Home Win", "Draw", "Away Win"][prediction]
        result_label.config(text=f"Predicted Result: {result_text}")

    def create_player_section(parent, title, team_key):
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(padx=10, pady=10)
        categories = ['Goalkeepers', 'Defenders', 'Midfielders', 'Forwards', 'Substitutes']
        listboxes = {}

        for i, cat in enumerate(categories):
            ttk.Label(frame, text=cat).grid(row=0, column=i)
            lb = tk.Listbox(frame, selectmode='multiple', height=10, exportselection=False)
            lb.grid(row=1, column=i, padx=5)
            lb.bind('<<ListboxSelect>>', update_counts)
            listboxes[cat] = lb

        ttk.Label(frame, textvariable=selected_counts[team_key], foreground="blue").grid(row=2, columnspan=5, pady=5)

        return listboxes

    # League selection
    ttk.Label(root, text="Select League:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    league_combo = ttk.Combobox(root, values=league_names, state="readonly")
    league_combo.grid(row=0, column=1, sticky="w")
    league_combo.bind("<<ComboboxSelected>>", update_team_selection)

    # Team selectors
    ttk.Label(root, text="Team 1 (Home):").grid(row=1, column=0, sticky="w")
    team1_combo = ttk.Combobox(root, state="readonly")
    team1_combo.grid(row=1, column=1, sticky="w")
    team1_combo.bind("<<ComboboxSelected>>", update_player_lists)

    ttk.Label(root, text="Team 2 (Away):").grid(row=2, column=0, sticky="w")
    team2_combo = ttk.Combobox(root, state="readonly")
    team2_combo.grid(row=2, column=1, sticky="w")
    team2_combo.bind("<<ComboboxSelected>>", update_player_lists)

    # Player sections
    players_team1_listboxes = create_player_section(root, "Players - Team 1", 'Team1')
    players_team2_listboxes = create_player_section(root, "Players - Team 2", 'Team2')

    # Predict button
    predict_button = ttk.Button(root, text="Predict Match", command=predict_match)
    predict_button.grid(row=5, column=0, columnspan=2, pady=20)

    result_label = ttk.Label(root, text="Predicted Result: ", font=("Arial", 14))
    result_label.grid(row=6, column=0, columnspan=2, pady=10)

    root.mainloop()
