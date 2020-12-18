from parameters import *
from PredictNewStateModel import predict_new_state
import math


def shield(sorted_actions, speed, state):
    for action in sorted_actions:
        if is_safe(action, speed, state):
            return action
        # else:
        #    print(str(action) + " is not safe!")

    # print("No safe action found. RIP")
    return 0

def is_safe(action, speed, state):
    distance, new_speed = predict_new_state(speed, action, BUFFER_TIME)
    new_x = (WIDTH / 2)
    new_y = (HEIGHT / 2) - int(distance * PIXELS_PER_METER)

    return check_safe_trajectory(int(new_x), int(new_y), new_speed, 0, state)

def check_safe_trajectory(x, y, new_speed, angle, state):
    distance = get_safe_distance_blocks(new_speed)

    x_angle = math.sin(math.radians(angle))
    y_angle = math.cos(math.radians(angle))

    current_distance = 0

    # if abs(x - WIDTH/2) < 8 and abs(y - HEIGHT/2) < 8:
    #    current_distance = 8

    #rgb = BirdViewProducer.as_rgb(self.state.transpose([2, 1, 0]))

    while current_distance < distance:
        block1 = dangerous_block(x + math.floor(x_angle * current_distance),
                                      y - math.floor(y_angle * current_distance), state)
        block2 = dangerous_block(x + math.floor(x_angle * current_distance),
                                      y - math.ceil(y_angle * current_distance), state)
        block3 = dangerous_block(x + math.ceil(x_angle * current_distance),
                                      y - math.floor(y_angle * current_distance), state)
        block4 = dangerous_block(x + math.ceil(x_angle * current_distance),
                                      y - math.ceil(y_angle * current_distance), state)

        #rgb[y - math.floor(y_angle * current_distance), x + math.floor(x_angle * current_distance)] = [0, 0, 0]
        #rgb[y - math.ceil(y_angle * current_distance), x + math.floor(x_angle * current_distance)] = [0, 0, 0]
        #rgb[y - math.floor(y_angle * current_distance), x + math.ceil(x_angle * current_distance)] = [0, 0, 0]
        #rgb[y - math.ceil(y_angle * current_distance), x + math.ceil(x_angle * current_distance)] = [0, 0, 0]

        if block1 or block2 or block3 or block4:
            return False

        current_distance += 1

    # plt.imshow(rgb)
    # plt.show()

    return True


def dangerous_block(x, y, state):
    if x < 0 or x > WIDTH or y < 0 or y > HEIGHT:
        # print("Out of state")
        return False

    if state[x, y, 4] == 1:
        # print("Safe: own car")
        return False
    elif state[x, y, 0] == 0:
        # print("Unsafe: no road")
        return True
    elif state[x, y, 3] == 1:
        # print("Unsafe: car")
        return True
    elif state[x, y, 8] == 1:
        # print("Unsafe: pedestrian")
        return True
    else:
        # print("Safe")
        return False


def get_safe_distance_blocks(speed):
    meters = 0

    while speed > 0.1:
        dist, speed = predict_new_state(speed, 0, 0.1)
        meters += dist

    return int(meters + BUFFER_DISTANCE) * PIXELS_PER_METER
