from parameters import *
from vel_to_acc import *


class shield:
    def __init__(self):
        self.vel_to_acc = VelToAcc()

    def shield(self, sorted_actions, speed, closest_object_distance):
        for action in sorted_actions:
            if self.is_safe(action, speed, closest_object_distance):
                return action
            # else:
            #    print(str(action) + " is not safe!")

        # print("No safe action found. RIP")
        return 0

    def is_safe(self, action, speed, closest_object_distance):
        new_speed = self.vel_to_acc.get_speed(action, speed, ACTION_TO_STATE_TIME)
        safe_distance = self.get_safe_distance(new_speed)

        if closest_object_distance < safe_distance + self.vel_to_acc.get_distance(action, speed, ACTION_TO_STATE_TIME):
            return False

        return True

    def get_safe_distance(self, speed):
        meters = 0.0

        while speed > 0.1:
            meters += self.vel_to_acc.get_distance(0, speed, ACTION_TO_STATE_TIME)
            speed = self.vel_to_acc.get_speed(0, speed, ACTION_TO_STATE_TIME)

        return meters + BUFFER_DISTANCE
