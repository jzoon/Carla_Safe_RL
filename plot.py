import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns
import pandas as pd



def plot_rewards(data):
    sns.lineplot(x=data.index, y="reward", data=data)
    plt.show()


def plot_collisions(collisions, distances):
    collisions_mean = np.mean(collisions, axis=0)
    distances_mean = np.mean(distances, axis=0)

    y = 1000*collisions_mean/distances_mean

    plot_data = gaussian_filter1d(y, sigma=5)
    plt.plot(plot_data)
    plt.show()


def plot_speed(times, distances):
    times_mean = np.mean(times, axis=0)
    distances_mean = np.mean(distances, axis=0)
    y = distances_mean/times_mean

    plot_data = gaussian_filter1d(y, sigma=5)
    plt.plot(plot_data)
    plt.show()


file_names = ["manual_logs/data_generation_1611771954.csv",
              "manual_logs/data_generation_1611829885.csv",
              "manual_logs/data_generation_1611830785.csv"]
distances = []
times = []
collisions = []
rewards = []
data = []

for filename in file_names:
    df = pd.read_csv(filename)
    data.append(df)

    #distances.append(data[:,0])
    #times.append(data[:,1])
    #collisions.append(data[:,2])
    #rewards.append(data[:,3])
pd.set_option("display.max_rows", None, "display.max_columns", None)

data = pd.concat(data)

plot_rewards(data)
#plot_collisions(np.array(collisions), np.array(distances))
#plot_speed(np.array(times), np.array(distances))
