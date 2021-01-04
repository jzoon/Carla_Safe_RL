from parameters import *
from PredictNewStateModel import predict_new_state


def shield(sorted_actions, speed, closest_object_distance):
    for action in sorted_actions:
        if is_safe(action, speed, closest_object_distance):
            return action
        # else:
        #    print(str(action) + " is not safe!")

    # print("No safe action found. RIP")
    return 0


def is_safe(action, speed, closest_object_distance):
    distance, new_speed = predict_new_state(speed, action, BUFFER_TIME)
    safe_distance = get_safe_distance(new_speed)

    if closest_object_distance < safe_distance + distance:
        return False

    return True


def get_safe_distance(speed):
    meters = 0

    while speed > 0.1:
        dist, speed = predict_new_state(speed, 0, 0.1)
        meters += dist

    return meters + BUFFER_DISTANCE
