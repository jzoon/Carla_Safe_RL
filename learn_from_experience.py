from numpy import genfromtxt
import re
from DQNAgent import *

replay = []
agent = DQNAgent()
MODEL_NAME = "exp"

with open("experiences/long_bug_fix_test_1611082068.csv", "r") as f:
    for line in f:
        l = line.split(",")
        state = [[float(l[0].strip("[")), float(l[1]), float(l[2].strip("]"))]]
        action = int(l[3])
        reward = float(l[4])
        new_state = [[float(l[5].strip("[")), float(l[6]), float(l[7].strip().strip("]"))]]

        agent.update_replay_memory((state, action, reward, new_state, False))

#agent = DQNAgent()
#agent.replay_memory = replay

agent.train_in_loop()
agent.model.save(f'models/{MODEL_NAME}__{int(time.time())}.model')