import json
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

player_data = pd.read_csv("weekly_points_data.csv")
player_data = player_data.head(400)
print(player_data.head())

#replacing values in datafram so it can be used for euclidean distance comparison
player_data.replace("-", 0, inplace=True)
player_data.replace("BYE", 0, inplace=True)
player_data.fillna(0, inplace=True)

#position values
position_dict = {
    "QB": 1,
    "RB": 2,
    "WR": 3,
    "TE": 4,
    "DST": 5,
    "K": 6
}

player_data['pos_num'] = player_data['Pos'].map(position_dict)

#team values
team_dict = {
    "WAS": 1, "DAL": 2, "PHI": 3, "NYG": 4,
    "CHI": 5, "DET": 6, "MIN": 7, "GB": 8,
    "NO": 9, "ATL": 10, "TB": 11, "CAR": 12,
    "SF": 13, "LAR": 14, "ARI": 15, "SEA": 16,
    "BAL": 17, "PIT": 18, "CIN": 19, "CLE": 20,
    "BUF": 21, "NE": 22, "MIA": 23, "NYJ": 24,
    "JAC": 25, "HOU": 26, "TEN": 27, "IND": 28,
    "KC": 29, "DEN": 30, "LAC": 31, "LV": 32,
    "FA": 33
}

player_data['team_num'] = player_data['Team'].map(team_dict)

player_data = player_data.drop("Pos", axis=1)
player_data = player_data.drop("Team", axis=1)

print(player_data.head())

# Select relevant features and target variable
features = ['pos_num', 'team_num', 'AVG', 'Week 1', 'Week 18']
target = 'TTL'

X = player_data[features]
y = player_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

#Prints the mean squared error and mean absolute error
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
