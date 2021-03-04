from parameters import *


class VelToAcc:
    def __init__(self, acc_actions):
        self.all_vel = []
        self.all_acc = []

        for action in acc_actions:
            vel, acc = self.read_file(str(action))
            self.all_vel.append(vel)
            self.all_acc.append(acc)

    def read_file(self, filename):
        vel = []
        acc = []

        with open("vel_acc_data/data_" + filename + ".txt", "r") as file:
            switch = False
            for line in file:
                line = line.strip()
                if line == "a":
                    switch = True
                elif switch:
                    acc.append(float(line))
                else:
                    vel.append(float(line))

        return vel, acc

    def get_acc(self, action, velocity):
        vel = self.all_vel[action]
        acc = self.all_acc[action]

        index = min(range(len(vel)), key=lambda i: abs(vel[i]-velocity))

        if index >= len(acc):
            index = len(acc) - 1

        return acc[index]

    def get_speed(self, action, velocity, time):
        acc = self.get_acc(action, velocity)
        new_vel = max(0, (velocity + (velocity + (acc * time))) / 2)
        if action > 6:
            new_vel += 0.45

        return new_vel

    def get_distance(self, action, velocity, time):
        new_velocity = self.get_speed(action, velocity, time)
        # This can overestimate the distance, since velocity might be 0 before time is over
        distance = ((new_velocity + velocity) / 2) * time

        return distance

ACC_ACTIONS = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
v = VelToAcc(ACC_ACTIONS)

import matplotlib.pyplot as plt
plt.tight_layout()

for i in reversed(range(11)):
    vel = [x for _,x in sorted(zip(v.all_vel[i],v.all_acc[i]))]
    plt.plot(sorted(v.all_vel[i]), vel, label=str(ACC_ACTIONS[i]))

plt.legend(bbox_to_anchor=(1, 1))
plt.show()
