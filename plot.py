import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns
import pandas as pd

names = ["0", "200", "500"]

def plot_rewards(all_data):
    for i, data in enumerate(all_data):
        sns.lineplot(x="episode", y="reward", data=data, label=names[i])
    plt.savefig("reward.png")
    plt.show()


def plot_collisions(all_data):
    for i, data in enumerate(all_data):
        sns.lineplot(x="episode", y="collisions per km", data=data, label=names[i])
    plt.savefig("col.png")
    plt.show()


def plot_speed(all_data):
    for i, data in enumerate(all_data):
        sns.lineplot(x="episode", y="speed", data=data, label=names[i])
    plt.savefig("speed.png")
    plt.show()


all_file_names = [["manual_logs/init_comp_0_1612867385.csv",
                "manual_logs/init_comp_0_1612877029.csv",
                "manual_logs/init_comp_0_1612886590.csv",
                "manual_logs/init_comp_0_1612896029.csv",
                "manual_logs/init_comp_0_1612906351.csv"],
                ["manual_logs/init_comp_200_1612916744.csv",
                "manual_logs/init_comp_200_1612929088.csv",
                "manual_logs/init_comp_200_1612940060.csv",
                "manual_logs/init_comp_200_1612950165.csv",
                "manual_logs/init_comp_200_1612960467.csv"],
                ["manual_logs/init_comp_500_1612976223.csv",
                "manual_logs/init_comp_500_1612997826.csv",
                "manual_logs/init_comp_500_1613008745.csv",
                "manual_logs/init_comp_500_1613018553.csv",
                "manual_logs/init_comp_500_1613028610.csv"]]

all_data = []

for file_names in all_file_names:

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

    all_data.append(data)

plot_rewards(all_data)
plot_collisions(all_data)
plot_speed(all_data)
