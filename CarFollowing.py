from parameters import *
import math


class CarFollowing:
    delta = 4
    d_0 = 2
    T = 1.6
    a = 3
    b = 8


    def get_action(self, velocity, distance, desired_vel, other_velocity):
        if distance == -1:
            distance = 10000
            other_velocity = 10000

        desired_acc = self.calculate_acceleration(velocity, distance, desired_vel, other_velocity)
        action = self.acceleration_to_action(desired_acc)

        return action

    def calculate_acceleration(self, velocity, distance, desired_vel, other_velocity):
        d_start = self.d_0 + self.T * velocity + ((velocity*(velocity-other_velocity))/(2*math.sqrt(self.a*self.b)))
        v = self.a * (1 - (velocity/desired_vel)**self.delta - (d_start/distance)**2)

        return v

    def acceleration_to_action(self, desired_acc):
        if desired_acc < -2:
            return 0
        elif desired_acc < -0.5:
            return 1
        elif desired_acc < 0.5:
            return 2
        elif desired_acc < 3:
            return 3

        return 4
