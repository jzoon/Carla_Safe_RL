from parameters import *
import spawn_npc
import math
import time
import sys
import glob
import os

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class CarEnv1:
    STATE_LENGTH = 1
    STATE_WIDTH = 2

    STEER_ACTIONS = [-0.5, 0.0, 0.5]
    ACC_ACTIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]

    actor_list = []
    collision_hist = []
    lane_hist = []
    distance = 0
    previous_location = None
    speed = 0
    acceleration = 0
    colsensor = None
    lanesensor = None
    episode_start = 0
    state = None
    location = None
    velocity = None
    transform = None
    obstacle = None
    left_road = False

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(4.0)

        self.world = self.client.load_world('Town01')
        self.world.set_weather(carla.WeatherParameters.ClearSunset)
        self.map = self.world.get_map()

        if not RENDERING:
            settings = self.world.get_settings()
            settings.no_rendering_mode = True
            self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter('model3')[0]
        self.model_3.set_attribute('color', '255,0,0')

        self.start_transform = self.world.get_map().get_spawn_points()[64]
        self.destination = self.world.get_map().get_spawn_points()[184] # VERANDEREN ZODAT DE AUTO NIET BOTST

        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(self.start_transform.location + carla.Location(z=50),
                                                carla.Rotation(pitch=-90)))

    def reset(self):
        self.distance = 0
        self.previous_location = None
        self.obstacle = None

        self.collision_hist = []
        self.actor_list = []
        self.lane_hist = []
        self.left_road = False
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
        time.sleep(0.5)

        velocity = carla.Vector3D(0, -INITIAL_SPEED, 0)
        self.vehicle.set_velocity(velocity)
        self.episode_start = time.time()

        self.update_parameters()

        return self.state

    def step(self, action_list):
        self.update_parameters()

        if SHIELD:
            action = self.shield(action_list)
        else:
            action = action_list[0]

        self.car_control(action)

        time.sleep(ACTION_TO_STATE_TIME)

        self.update_parameters()
        self.update_KPIs(self.location)
        reward, done = self.get_reward_and_done()

        return self.state, reward, done, action

    def update_parameters(self):
        self.location = self.vehicle.get_location()
        self.transform = self.vehicle.get_transform()
        self.velocity = self.vehicle.get_velocity()
        self.speed = math.sqrt(self.velocity.x ** 2 + self.velocity.y ** 2 + self.velocity.z ** 2)
        self.state = self.get_state()

    def get_reward_and_done(self):
        reward = self.reward_simple()

        if self.passed_destination(self.location, self.previous_location) or self.episode_start + SECONDS_PER_EPISODE < time.time():
            return reward, True

        if len(self.lane_hist) != 0:
            return -SIMPLE_REWARD_B, True

        if len(self.collision_hist) != 0:
            return -SIMPLE_REWARD_B, True

        return reward, False

    def reward_simple(self):
        v_max = self.vehicle.get_speed_limit()
        if v_max < 1:
            v_max = 50

        return SIMPLE_REWARD_A * (self.speed / v_max)

    def car_control(self, action):
        steer_action = int(action / len(self.ACC_ACTIONS))
        acc_action = action % len(self.ACC_ACTIONS)

        if self.ACC_ACTIONS[acc_action] < 0:
            self.vehicle.apply_control(
                carla.VehicleControl(brake=-self.ACC_ACTIONS[acc_action], steer=self.STEER_ACTIONS[steer_action])
            )
        else:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=self.ACC_ACTIONS[acc_action], steer=self.STEER_ACTIONS[steer_action])
            )

    def update_KPIs(self, current_location):
        if self.previous_location is None:
            self.previous_location = current_location
        else:
            self.distance += self.calculate_distance(current_location, self.previous_location)

        self.previous_location = current_location

    def passed_destination(self, current_location, previous_location):
        up_x = max(current_location.x, previous_location.x) + 1
        down_x = min(current_location.x, previous_location.x) - 1
        up_y = max(current_location.y, previous_location.y) + 1
        down_y = min(current_location.y, previous_location.y) - 1

        if down_x < self.destination.location.x < up_x and down_y < self.destination.location.y < up_y:
            return True

        return False

    def get_state(self):
        return [[self.transform.rotation.yaw, self.map.get_waypoint(self.location).transform.rotation.yaw]]

    def get_KPI(self):
        return self.calculate_distance(self.location, self.start_transform.location), len(self.collision_hist) > 0, len(self.lane_hist) > 0

    def calculate_distance(self, location_a, location_b):
        return math.sqrt((location_a.x - location_b.x) ** 2 + (location_a.y - location_b.y) ** 2)

    def car_following(self, _):
        desired_velocity = self.vehicle.get_speed_limit() * 0.95

        if self.speed < desired_velocity:
            return 9
        else:
            return 7

    def shield(self, action_list):
        for action in action_list:
            if self.STEER_ACTIONS[int(action / len(self.ACC_ACTIONS))] == 0.0:
                return action

        return 5