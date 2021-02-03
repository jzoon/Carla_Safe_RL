import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns
import pandas as pd

names = ["DDQN", "Shield", "SIP", "Both"]

def plot_rewards(all_data):
    for i, data in enumerate(all_data):
        sns.lineplot(x="episode", y="reward", data=data, label=names[i])
    plt.savefig("reward.png")


def plot_collisions(all_data):
    for i, data in enumerate(all_data):
        sns.lineplot(x="episode", y="collisions per km", data=data, label=names[i])
    plt.savefig("col.png")


def plot_speed(all_data):
    for i, data in enumerate(all_data):
        sns.lineplot(x="episode", y="speed", data=data, label=names[i])
    plt.savefig("speed.png")


all_file_names = [["manual_logs/new_data_generation_1611858073.csv",
                "manual_logs/new_data_generation_1611865669.csv",
                "manual_logs/new_data_generation_1611872985.csv",
                "manual_logs/new_data_generation_1611880164.csv",
                "manual_logs/new_data_generation_1611887415.csv"],
                ["manual_logs/shield_data_generation_1611916856.csv",
                "manual_logs/shield_data_generation_1611926163.csv",
                "manual_logs/shield_data_generation_1611935809.csv",
                "manual_logs/shield_data_generation_1611945387.csv",
                "manual_logs/shield_data_generation_1611955527.csv"],
                ["manual_logs/policy_data_generation_1611964355.csv",
                "manual_logs/policy_data_generation_1611971578.csv",
                "manual_logs/policy_data_generation_1611980929.csv",
                "manual_logs/policy_data_generation_1611990152.csv",
                "manual_logs/policy_data_generation_1611998625.csv"],
                ["manual_logs/both_data_generation_1612059650.csv",
                "manual_logs/both_data_generation_1612069007.csv",
                "manual_logs/both_data_generation_1612079122.csv",
                "manual_logs/both_data_generation_1612088585.csv",
                "manual_logs/both_data_generation_1612098663.csv"]]

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

#plot_rewards(all_data)
#plot_collisions(all_data)
plot_speed(all_data)
