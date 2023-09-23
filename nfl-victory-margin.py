# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 12:52:08 2023

@author: Local User
"""

import requests
import pandas as pd
import numpy as np
import time
import sqlalchemy
import mysql.connector
import meteostat
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report

dates = pd.date_range(start = "2019-09-07", end = "2023-02-28").strftime("%Y%m%d")

victory_margins = []

for date in dates:
    
    try:

        available_games = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={date}").json()["events"]
        
    except Exception:
        continue
    
    if len(available_games) < 1:
        continue

    for game in available_games:
        
        if "AFC" in game["name"]:
            continue
        
        game_id = int(game["id"])
        game_date = pd.to_datetime(game["date"]).tz_convert("US/Eastern")
        
        play_by_play_data = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={game_id}").json()
        
        if "winprobability" not in play_by_play_data:
            continue
        
        home_win_percentage = play_by_play_data["winprobability"][0]["homeWinPercentage"]
        away_win_percentage = (1 - home_win_percentage)
        
        game_selection = game["competitions"][0]["competitors"]
        
        # team 1
        
        team_1_name = game_selection[0]["team"]["displayName"]
        team_1_side = game_selection[0]["homeAway"]
        
        if team_1_side == "home":
            
            team_1_win_percentage = home_win_percentage
            
        elif team_1_side == "away":
            
            team_1_win_percentage = away_win_percentage
        
        team_1_score = int(game_selection[0]["score"])
        
        # team 2
        
        team_2_name = game_selection[1]["team"]["displayName"]
        team_2_side = game_selection[1]["homeAway"]
        
        if team_2_side == "home":
            
            team_2_win_percentage = home_win_percentage
            
        elif team_2_side == "away":
            
            team_2_win_percentage = away_win_percentage
        
        team_2_score = int(game_selection[1]["score"])
        
        # designate favorite
        
        if team_1_win_percentage > team_2_win_percentage:
            
            favorite = "team_1"
            
        elif team_1_win_percentage < team_2_win_percentage:
            
            favorite = "team_2"
            
        # designate winner
            
        if team_1_score > team_2_score:
            
            winner = "team_1"
            
        elif team_1_score < team_2_score:
            
            winner = "team_2"
        
        victory_margin = abs(team_1_score - team_2_score)
        
        victory_margin_dataframe = pd.DataFrame([{"game_date":game_date,"team_1_name": team_1_name, "team_1_score": team_1_score, "team_2_name":team_2_name, "team_2_score":team_2_score, "victory_margin":victory_margin, "favorite":favorite, "winner":winner}])
        
        victory_margins.append(victory_margin_dataframe)
        
total_victory_margins = pd.concat(victory_margins).set_index("game_date")

game_team_1 = "Los Angeles Chargers"
game_team_2 = "New Orleans Saints"

# get all games containing either team
team_victory_margins = total_victory_margins[(total_victory_margins["team_1_name"] == game_team_1) | (total_victory_margins["team_2_name"] == game_team_1) | (total_victory_margins["team_1_name"] == game_team_2) | (total_victory_margins["team_2_name"] == game_team_2)]
matchup_victory_margins = total_victory_margins[(total_victory_margins["team_1_name"] == game_team_1) & (total_victory_margins["team_2_name"] == game_team_2) | (total_victory_margins["team_1_name"] == game_team_2) & (total_victory_margins["team_2_name"] == game_team_1)]

# get upsets
upsets = team_victory_margins[team_victory_margins["favorite"] != team_victory_margins["winner"]]

# upsets where the current under dog won | upsets where the current favorite lost
replicated_favorite_upsets = upsets[(upsets["team_1_name"] == game_team_1) & (upsets["favorite"] == "team_2") | (upsets["team_2_name"] == game_team_1) & (upsets["favorite"] == "team_1") | (upsets["team_1_name"] == game_team_2) & (upsets["favorite"] == "team_1") | (upsets["team_2_name"] == game_team_2) & (upsets["favorite"] == "team_2")]

data = replicated_favorite_upsets["victory_margin"]
mean = np.mean(data)
std_dev = np.std(data)

plt.figure(dpi = 800)

plt.hist(data, bins=20, density=True, alpha=0.6, color='b', label='Histogram')

x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
pdf = (1/(std_dev * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mean)/std_dev)**2)

plt.plot(x, pdf, color='r', label='Normal Distribution')
plt.axvline(mean, color='k', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(mean - std_dev, color='g', linestyle='dashed', linewidth=1, label='1 STD Dev')
plt.axvline(mean + std_dev, color='g', linestyle='dashed', linewidth=1)
plt.axvline(mean - 2*std_dev, color='y', linestyle='dashed', linewidth=1, label='2 STD Dev')
plt.axvline(mean + 2*std_dev, color='y', linestyle='dashed', linewidth=1)
plt.axvline(mean - 3*std_dev, color='b', linestyle='dashed', linewidth=1, label='3 STD Dev')
plt.axvline(mean + 3*std_dev, color='b', linestyle='dashed', linewidth=1)

plt.xlabel('Victory Margin')
plt.ylabel('Probability Density')
plt.title('Victory Margin of Games Where Lions were Underdog or Chiefs were Favorite')
plt.suptitle("Upsets Only")
plt.suptitle(f"Mean: {mean}")
plt.legend()

plt.show()

pre_transformed_dataset = replicated_favorite_upsets.drop(["team_1_score", "team_2_score", "winner"], axis = 1)
pre_transformed_dataset["year"] = pre_transformed_dataset.index.year
pre_transformed_dataset["month"] = pre_transformed_dataset.index.month
pre_transformed_dataset["day"] = pre_transformed_dataset.index.day
pre_transformed_dataset = pre_transformed_dataset.reset_index(drop = True)

game_production_dataset = pd.DataFrame([{"team_1_name": game_team_1, "team_2_name":game_team_2, "favorite": game_team_2, "year":2023, "month":8, "day":20}])

updated_pre_transformed_dataset = pd.concat([pre_transformed_dataset, game_production_dataset], axis = 0).reset_index(drop = True)

transformed_dataset = pd.get_dummies(updated_pre_transformed_dataset)

X = transformed_dataset.head(len(transformed_dataset)-1).drop("victory_margin", axis = 1)
Y = transformed_dataset.head(len(transformed_dataset)-1)["victory_margin"]

base_model = RandomForestClassifier(n_estimators=100)
fitted_model = base_model.fit(X = X, y = Y)

transformed_game_production_dataset = transformed_dataset.tail(1).drop("victory_margin", axis = 1)

model_prediction = fitted_model.predict(X = transformed_game_production_dataset)

#####


engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')

total_victory_margins.to_sql("nfl_victory_margins", con = engine, if_exists = "append")

# If you make a mistake, or wish to re-build the dataset, you can drop the table and start over

with engine.connect() as conn:
    result = conn.execute(sqlalchemy.text('DROP TABLE nfl_victory_margins'))