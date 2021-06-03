from parameters import *
import math
from vel_to_acc import *


# This class is based on the Intelligent Driver Model (Treiber et al.) and determines the desired acceleration and
# corresponding action to follow the leading vehicle based on this model.
class CarFollowing:
    delta = 4
    d_0 = 30
    T = 4
    a = 3
    b = 8

    def __init__(self, acc_actions):
        self.acc_actions = acc_actions
        self.vel_to_acc = VelToAcc(acc_actions)

    def get_action(self, velocity, distance, desired_vel, other_velocity):
        if distance == -1:
            distance = 10000
            other_velocity = 10000

        desired_acc = self.calculate_acceleration(velocity, distance, desired_vel, other_velocity)
        action = self.acceleration_to_action(velocity, desired_acc)

        return action

    def calculate_acceleration(self, velocity, distance, desired_vel, other_velocity):
        if distance == 0:
            distance = 1
        if desired_vel == 0:
            desired_vel = 1

        d_start = self.d_0 + self.T * velocity + ((velocity*(velocity-other_velocity))/(2*math.sqrt(self.a*self.b)))
        v = self.a * (1 - (velocity/desired_vel)**self.delta - (d_start/distance)**2)

        return v

    def acceleration_to_action(self, velocity, desired_acc):
        for action in reversed(range(len(self.acc_actions))):
            if desired_acc > self.vel_to_acc.get_acc(action, velocity):
                return action

        return 0
