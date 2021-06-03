import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

figure_name = "Scenario1_"
names = ["DDQN", "SCS", "SIPS"]

save = True

def plot_rewards(all_data):
    for i, data in enumerate(all_data):
        sns.lineplot(x="episode", y="return", data=data, label=names[i], ci="sd")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend(loc='upper left')
    if save:
        plt.savefig("plots/pdf/" + figure_name + "return.pdf")
    plt.show()


def plot_collisions(all_data):
    for i, data in enumerate(all_data):
        sns.lineplot(x="episode", y="collisions per km", data=data, label=names[i], ci="sd")
    plt.xlabel("Episode")
    plt.ylabel("Collisions per km")
    plt.legend(loc='upper right')
    if save:
        plt.savefig("plots/pdf/" + figure_name + "col.pdf")
    plt.show()


def plot_speed(all_data):
    for i, data in enumerate(all_data):
        sns.lineplot(x="episode", y="speed", data=data, label=names[i], ci="sd")
    plt.xlabel("Episode")
    plt.ylabel("Average speed (m/s)")
    plt.legend(loc='upper left')
    if save:
        plt.savefig("plots/pdf/" + figure_name + "speed.pdf")
    plt.show()

def plot_overrule(all_data):
    for i, data in enumerate(all_data):
        sns.lineplot(x="episode", y="overrule", data=data, label=names[i], ci="sd")
    plt.xlabel("Episode")
    plt.ylabel("Overruled actions by shield (%)")
    plt.legend(loc='upper right')
    if save:
        plt.savefig("plots/pdf/" + figure_name + "overrule.pdf")
    plt.show()

all_file_names = [["manual_logs/Scenario1/DDQN/Scenario1_Shield0_SIPshield0--1616658124.csv",
"manual_logs/Scenario1/DDQN/Scenario1_Shield0_SIPshield0--1616661125.csv",
"manual_logs/Scenario1/DDQN/Scenario1_Shield0_SIPshield0--1616664330.csv",
"manual_logs/Scenario1/DDQN/Scenario1_Shield0_SIPshield0--1616677881.csv",
"manual_logs/Scenario1/DDQN/Scenario1_Shield0_SIPshield0--1616681009.csv"],
["manual_logs/Scenario1/SCS/Scenario1_Shield1_SIPshield0--1616581639.csv",
"manual_logs/Scenario1/SCS/Scenario1_Shield1_SIPshield0--1616585968.csv",
"manual_logs/Scenario1/SCS/Scenario1_Shield1_SIPshield0--1616590225.csv",
"manual_logs/Scenario1/SCS/Scenario1_Shield1_SIPshield0--1616594652.csv",
"manual_logs/Scenario1/SCS/Scenario1_Shield1_SIPshield0--1616598888.csv"],
["manual_logs/Scenario1/SIPS/Scenario1_Shield0_SIPshield1--1618392515.csv",
"manual_logs/Scenario1/SIPS/Scenario1_Shield0_SIPshield1--1618396840.csv",
"manual_logs/Scenario1/SIPS/Scenario1_Shield0_SIPshield1--1618401265.csv",
"manual_logs/Scenario1/SIPS/Scenario1_Shield0_SIPshield1--1618405611.csv",
"manual_logs/Scenario1/SIPS/Scenario1_Shield0_SIPshield1--1618410113.csv"]
]

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
plot_overrule(all_data)
