from parameters import *
from vel_to_acc import *
import random


class shield:
    def __init__(self, acc_actions):
        self.vel_to_acc = VelToAcc(acc_actions)

    def shield(self, sorted_actions, speed, closest_object_distance):
        for action in sorted_actions:

            if self.is_safe(action, speed, closest_object_distance):
                return action

        return 0

    def is_safe(self, action, speed, closest_object_distance):
        new_speed = self.vel_to_acc.get_speed(action, speed, ACTION_TO_STATE_TIME)
        distance_driven = self.vel_to_acc.get_distance(action, speed, ACTION_TO_STATE_TIME)
        safe_distance = self.get_safe_distance(new_speed)

        if closest_object_distance < safe_distance + distance_driven:
            return False

        return True

    def get_safe_distance(self, speed):
        min_deceleration = -self.vel_to_acc.get_acc(0, 1)
        brake_distance = (0.5*speed**2)/min_deceleration

        return brake_distance + BUFFER_DISTANCE
