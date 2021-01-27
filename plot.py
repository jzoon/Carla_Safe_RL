import numpy as np
import matplotlib.pyplot as plt


def plot_rewards():
    N = 100
    plot_data = np.convolve(data[0][:,3], np.ones(N) / N, mode='valid')
    plt.plot(plot_data)
    plt.show()


def plot_collisions():
    N = 100
    #plot_data = np.convolve(1000*data[0][:,2]/data[0][:,0], np.ones(N) / N, mode='valid')
    plt.plot(1000*data[0][:,2]/data[0][:,0])
    plt.show()


def plot_speed():
    N = 100
    plot_data = np.convolve(data[0][:,0]/data[0][:,1], np.ones(N) / N, mode='valid')
    plt.plot(plot_data)
    plt.show()


file_names = ["manual_logs/data_test_1611758232.csv"]
data = []

for filename in file_names:
    data.append(np.genfromtxt(filename, delimiter=','))

plot_rewards()
plot_collisions()
plot_speed()
