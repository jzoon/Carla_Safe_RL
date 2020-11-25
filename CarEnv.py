from parameters import *
import spawn_npc
import math
import time
import sys
import glob
import os
import matplotlib.pyplot as plt

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions


class CarEnv:
    actor_list = []
    collision_hist = []
    lane_hist = []
    distance = 0
    wrong_steps = 0
    previous_location = None
    speed = 0
    colsensor = None
    lanesensor = None
    episode_start = 0
    state = None
    location = None
    previous_distance_to_destination = 0
    transform = None

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(4.0)

        self.world = self.client.load_world('Town01')
        self.world.set_weather(carla.WeatherParameters.ClearSunset)

        if not RENDERING:
            settings = self.world.get_settings()
            settings.no_rendering_mode = True
            self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter('model3')[0]

        self.start_transform = self.world.get_map().get_spawn_points()[2]
        self.destination = self.world.get_map().get_spawn_points()[2]
        self.destination.location.x -= DESTINATION_DISTANCE

        self.birdview_producer = BirdViewProducer(
            self.destination,
            self.client,
            target_size=PixelDimensions(width=WIDTH, height=HEIGHT),
            pixels_per_meter=PIXELS_PER_METER,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA)

        if OTHER_TRAFFIC:
            spawn_npc.main()

    def reset(self):
        self.distance = 0
        self.wrong_steps = 0
        self.previous_location = None

        self.collision_hist = []
        self.actor_list = []
        self.lane_hist = []

        self.vehicle = None

        while self.vehicle is None:
            self.vehicle = self.world.try_spawn_actor(self.model_3, self.start_transform)

        self.actor_list.append(self.vehicle)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        sensor_transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        colsensor = self.blueprint_library.find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_hist.append(event))

        lanesensor = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lanesensor = self.world.spawn_actor(lanesensor, sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.lanesensor)
        self.lanesensor.listen(lambda event: self.lane_hist.append(event))

        self.previous_distance_to_destination = self.calculate_distance(self.destination.location.x, self.start_transform.location.x, self.destination.location.y, self.start_transform.location.y)

        self.birdview_producer.produce(agent_vehicle=self.vehicle)
        self.episode_start = time.time()

        return self.get_state()

    def step(self, action):
        self.car_control(action)
        self.state = self.get_state()
        self.transform = self.vehicle.get_transform()
        self.location = self.vehicle.get_location()
        self.update_KPIs(self.location)
        v = self.vehicle.get_velocity()
        self.speed = int(math.sqrt(v.x**2 + v.y**2 + v.z**2))

        reward, done = self.get_reward_and_done()

        return self.state, reward, done, None

    def get_reward_and_done(self):
        reward = 0
        done = False

        if self.wrong_location() != 0:
            reward -= 10

        reward += int(self.speed / 2)

        reward -= len(self.lane_hist) * 10
        self.lane_hist = []

        dist_to_dest = self.calculate_distance(self.location.x, self.destination.location.x, self.location.y,
                                               self.destination.location.y)
        if self.passed_destination(self.location, self.previous_location):
            reward = 200
            done = True
            self.previous_distance_to_destination = 0
        elif dist_to_dest < self.previous_distance_to_destination:
            reward += 2 * (self.previous_distance_to_destination - dist_to_dest)
            self.previous_distance_to_destination = dist_to_dest

        if len(self.collision_hist) != 0:
            done = True
            reward = -200

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return reward, done

    def car_control(self, action):
        steer_action = int(action / len(STEER_ACTIONS))
        acc_action = action % len(ACC_ACTIONS)

        if ACC_ACTIONS[acc_action] < 0:
            self.vehicle.apply_control(
                carla.VehicleControl(brake=-ACC_ACTIONS[acc_action], steer=STEER_ACTIONS[steer_action])
            )
        else:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=ACC_ACTIONS[acc_action], steer=STEER_ACTIONS[steer_action])
            )

    def update_KPIs(self, current_location):
        if self.previous_location is None:
            self.previous_location = current_location
        else:
            self.distance += self.calculate_distance(current_location.x, self.previous_location.x, current_location.y, self.previous_location.y)

        if self.wrong_location() > 0:
            self.wrong_steps += 1

    def wrong_location(self):
        dif = abs(abs(self.transform.rotation.yaw) - self.world.get_map().get_waypoint(self.location).transform.rotation.yaw)

        if 90 <= dif <= 270:
            return 1
        elif self.world.get_map().get_waypoint(self.location, project_to_road=False,
                                               lane_type=carla.LaneType.Sidewalk) is not None:
            return 2

        return 0

    def passed_destination(self, current_location, previous_location):
        up_x = max(current_location.x, previous_location.x) + 1
        down_x = min(current_location.x, previous_location.x) - 1
        up_y = max(current_location.y, previous_location.y) + 1
        down_y = min(current_location.y, previous_location.y) - 1

        if down_x < self.destination.location.x < up_x and down_y < self.destination.location.y < up_y:
            return True

        return False

    def get_state(self):
        return self.birdview_producer.produce(agent_vehicle=self.vehicle).transpose((2, 1, 0))

    def get_KPI(self):
        return self.distance, len(self.collision_hist) > 0, self.wrong_steps, self.previous_distance_to_destination

    def get_speed_limit(self):
        return self.vehicle.get_speed_limit()*3.6

    def calculate_distance(self, x1, x2, y1, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def shield(self, sorted_actions):
        for action in sorted_actions:
            if self.is_safe(action):
                return action

        print("No safe action found. RIP")
        return -1

    def is_safe(self, action):
        x_dif, y_dif, relative_angle = self.get_new_transform(action)
        new_x = (WIDTH/2) + x_dif
        new_y = (HEIGHT/2) - y_dif

        return self.check_safe_trajectory(int(new_x), int(new_y), relative_angle)

    def check_safe_trajectory(self, x, y, angle):
        distance = self.get_safe_distance_blocks()

        x_angle = math.sin(math.radians(angle))
        y_angle = math.cos(math.radians(angle))

        current_distance = 0

        if abs(x - WIDTH/2) < 12 and abs(y - HEIGHT/2) < 12:
            current_distance = 12

        rgb = BirdViewProducer.as_rgb(self.state.transpose([2,1,0]))

        while current_distance < distance * PIXELS_PER_METER:
            block1 = self.dangerous_block(x + math.floor(x_angle*current_distance), y - math.floor(y_angle*current_distance))
            block2 = self.dangerous_block(x + math.floor(x_angle*current_distance), y - math.ceil(y_angle*current_distance))
            block3 = self.dangerous_block(x + math.ceil(x_angle * current_distance), y - math.floor(y_angle * current_distance))
            block4 = self.dangerous_block(x + math.ceil(x_angle * current_distance), y - math.ceil(y_angle * current_distance))

            rgb[y - math.floor(y_angle*current_distance), x + math.floor(x_angle*current_distance)] = [0,0,0]
            rgb[y - math.ceil(y_angle * current_distance), x + math.floor(x_angle * current_distance)] = [0, 0, 0]
            rgb[y - math.floor(y_angle * current_distance), x + math.ceil(x_angle * current_distance)] = [0, 0, 0]
            rgb[y - math.ceil(y_angle * current_distance), x + math.ceil(x_angle * current_distance)] = [0, 0, 0]

            if block1 or block2 or block3 or block4:
                plt.imshow(rgb)
                plt.show()
                return False

            current_distance += 1

        plt.imshow(rgb)
        plt.show()

        return True

    def dangerous_block(self, x, y):
        if x < 0 or x > WIDTH or y < 0 or y > HEIGHT:
            #print("Out of state")
            return False

        if self.state[x, y, 4] == 1:
            #print("Safe: own car")
            return False
        elif self.state[x, y, 0] == 0:
            print("Unsafe: no road")
            return True
        elif self.state[x, y, 3] == 1:
            print("Unsafe: car")
            return True
        elif self.state[x, y, 8] == 1:
            print("Unsafe: pedestrian")
            return True
        else:
            #print("Safe")
            return False

    def get_new_transform(self, action):
        acc_action = action % len(ACC_ACTIONS)

        if acc_action < 2:
            acc_action = ACC_ACTIONS[acc_action] * BRAKE_POWER
        elif acc_action > 2:
            acc_action = ACC_ACTIONS[acc_action] * ACC_POWER

        x_dif = 0
        y_dif = int(((self.speed + BUFFER_TIME * acc_action * 0.5) * BUFFER_TIME) * PIXELS_PER_METER)

        return x_dif, y_dif, 0

    def get_safe_distance_blocks(self):
        return (0.5 * self.speed**2) / BRAKE_POWER
