import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns
import pandas as pd

figure_name = "Scenario1_DDQN_"
names = ["DDQN", "SCS", "SIPS"]

def plot_rewards(all_data):
    for i, data in enumerate(all_data):
        sns.lineplot(x="episode", y="return", data=data, label=names[i], ci="sd")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    #plt.savefig("plots/" + figure_name + "return.png")
    plt.show()


def plot_collisions(all_data):
    for i, data in enumerate(all_data):
        sns.lineplot(x="episode", y="collisions per km", data=data, label=names[i], ci="sd")
    plt.xlabel("Episode")
    plt.ylabel("Collisions per km")
    #plt.savefig("plots/" + figure_name + "col.png")
    plt.show()


def plot_speed(all_data):
    for i, data in enumerate(all_data):
        sns.lineplot(x="episode", y="speed", data=data, label=names[i], ci="sd")
    plt.xlabel("Episode")
    plt.ylabel("Average speed (m/s)")
    #plt.savefig("plots/" + figure_name + "speed.png")
    plt.show()

def plot_overrule(all_data):
    for i, data in enumerate(all_data):
        sns.lineplot(x="episode", y="overrule", data=data, label=names[i], ci="sd")
    plt.xlabel("Episode")
    plt.ylabel("Overruled actions by shield (%)")
    plt.savefig("plots/" + figure_name + "overrule.png")
    plt.show()

all_file_names = [["manual_logs/Scenario1_Shield0_SIPshield0--1616658124.csv",
"manual_logs/Scenario1_Shield0_SIPshield0--1616661125.csv",
"manual_logs/Scenario1_Shield0_SIPshield0--1616664330.csv",
"manual_logs/Scenario1_Shield0_SIPshield0--1616677881.csv",
"manual_logs/Scenario1_Shield0_SIPshield0--1616681009.csv"]]
#["manual_logs/Scenario1_Shield1_SIPshield0--1616581639.csv",
#"manual_logs/Scenario1_Shield1_SIPshield0--1616585968.csv",
#"manual_logs/Scenario1_Shield1_SIPshield0--1616590225.csv",
#"manual_logs/Scenario1_Shield1_SIPshield0--1616594652.csv",
#"manual_logs/Scenario1_Shield1_SIPshield0--1616598888.csv"]]
all_file_names = [["manual_logs/Scenario2_Shield0_SIPshield0--1615899395.csv"]]


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
    data["overrule"] = data["overrule"]*100

    all_data.append(data)

plot_rewards(all_data)
plot_collisions(all_data)
plot_speed(all_data)
#plot_overrule(all_data)
