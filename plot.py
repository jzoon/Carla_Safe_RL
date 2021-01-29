import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns
import pandas as pd



def plot_rewards(data):
    sns.lineplot(x="episode", y="reward", data=data)
    plt.show()


def plot_collisions(data):
    sns.lineplot(x="episode", y="collisions per km", data=data)
    plt.show()


def plot_speed(data):
    sns.lineplot(x="episode", y="speed", data=data)
    plt.show()


file_names = ["manual_logs/new_data_generation_1611858073.csv",
            "manual_logs/new_data_generation_1611865669.csv",
            "manual_logs/new_data_generation_1611872985.csv",
            "manual_logs/new_data_generation_1611880164.csv",
            "manual_logs/new_data_generation_1611887415.csv",
            "manual_logs/new_data_generation_1611895317.csv"]
distances = []
times = []
collisions = []
rewards = []
data = []

for filename in file_names:
    df = pd.read_csv(filename)
    data.append(df)

data = pd.concat(data)

data["collisions per km"] = 1000*data["collision"]/data["distance"]
data["speed"] = data["distance"]/data["time"]

plot_rewards(data)
plot_collisions(data)
plot_speed(data)
