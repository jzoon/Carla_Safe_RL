from numpy import genfromtxt
import re
from DQNAgent import *
from vel_to_acc import *
import matplotlib.pyplot as plt

replay = []
#agent = DQNAgent()
MODEL_NAME = "exp"

mistakes = [[], [], [], [], [], [], [], [], [], [], []]
va = VelToAcc([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

with open("experiences/test_more_actions_1614262099.csv", "r") as f:
    for line in f:
        l = line.split(",")
        state = [[float(l[0].strip("[")), float(l[1]), float(l[2].strip("]"))]]
        action = int(l[3])
        reward = float(l[4])
        new_state = [[float(l[5].strip("[")), float(l[6]), float(l[7].strip().strip("]"))]]
        if reward >= 0:
            if abs(va.get_speed(action, state[0][0], 2) - new_state[0][0]) > 3:
                print(line)
            mistakes[action].append(va.get_speed(action, state[0][0], 2) - new_state[0][0])

        #agent.update_replay_memory((state, action, reward, new_state, False))

#agent = DQNAgent()
#agent.replay_memory = replay

#agent.train_in_loop()
#agent.model.save(f'models/{MODEL_NAME}__{int(time.time())}.model')

plt.boxplot(mistakes)
plt.show()
for i in range(11):
    print(np.median(mistakes[i]))