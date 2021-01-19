from parameters import *
import spawn_npc
import math
import time
import sys
import glob
import os
import matplotlib.pyplot as plt
from PredictNewStateModel import predict_new_state
import random
from CarFollowing import *
from Shield import shield
import numpy as np

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
    acceleration = 0
    colsensor = None
    lanesensor = None
    episode_start = 0
    state = None
    location = None
    previous_distance_to_destination = 0
    transform = None
    obstacle = None

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

        self.start_transform = self.world.get_map().get_spawn_points()[2]
        self.destination = self.world.get_map().get_spawn_points()[2]
        self.destination.location.x -= DESTINATION_DISTANCE

        if OTHER_TRAFFIC:
            spawn_npc.main()

    def reset(self):
        self.distance = 0
        self.wrong_steps = 0
        self.previous_location = None
        self.obstacle = None

        self.collision_hist = []
        self.actor_list = []
        self.lane_hist = []

        self.vehicle = None

        while self.vehicle is None:
            self.vehicle = self.world.try_spawn_actor(self.model_3, self.start_transform)

        #self.start_transform.location.x -= 50
        #asfd = self.world.spawn_actor(self.model_3, self.start_transform)
        #self.start_transform.location.x += 50
        #self.actor_list.append(asfd)

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

        obstacle_sensor = self.blueprint_library.find('sensor.other.obstacle')
        obstacle_sensor.set_attribute('only_dynamics', 'True')
        obstacle_sensor.set_attribute('distance', '200')
        self.obstacle_sensor = self.world.spawn_actor(obstacle_sensor, sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.obstacle_sensor)
        self.obstacle_sensor.listen(lambda event: self.append_obstacle(event))

        self.previous_distance_to_destination = self.calculate_distance(self.destination.location, self.start_transform.location)

        time.sleep(0.5)
        if self.obstacle is not None:
            while self.calculate_distance(self.vehicle.get_location(), self.obstacle.other_actor.get_location()) < 15:
                time.sleep(0.1)

        velocity = carla.Vector3D(-INITIAL_SPEED, 0, 0)
        self.vehicle.set_velocity(velocity)
        self.episode_start = time.time()
        self.location = self.vehicle.get_location()
        self.transform = self.vehicle.get_transform()
        self.velocity = self.vehicle.get_velocity()
        self.state = self.get_state()

        return self.state

    def step(self, action_list):
        self.transform = self.vehicle.get_transform()
        self.location = self.transform.location
        self.velocity = self.vehicle.get_velocity()
        self.speed = math.sqrt(self.velocity.x ** 2 + self.velocity.y ** 2 + self.velocity.z ** 2)
        a = self.vehicle.get_acceleration()
        self.acceleration = int(math.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2))

        self.state = self.get_state()

        if SHIELD:
            action = self.shield(action_list)
        else:
            action = action_list[0]

        self.car_control(action)

        time.sleep(ACTION_TO_STATE_TIME)

        self.transform = self.vehicle.get_transform()
        self.location = self.transform.location
        v = self.vehicle.get_velocity()
        self.speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        self.state = self.get_state()
        self.update_KPIs(self.location)
        reward, done = self.get_reward_and_done(action, action_list)

        vehicle_list = self.world.get_actors().filter("vehicle.*")

        if len(vehicle_list) < 40:
            spawn_npc.main()

        return self.state, reward, done, action

    def get_reward_and_done(self, action, action_list):
        done = False

        reward = self.reward_simple()

        if len(self.collision_hist) != 0:
            done = True
            reward -= SIMPLE_REWARD_B

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        if self.passed_destination(self.location, self.previous_location):
            done = True

        return reward, done

    def reward_simple(self):
        v_max = self.vehicle.get_speed_limit()
        if v_max < 1:
            v_max = 50

        return SIMPLE_REWARD_A * (self.speed / v_max)

    def car_control(self, action):
        steer_action = int(action / len(ACC_ACTIONS))
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
            self.distance += self.calculate_distance(current_location, self.previous_location)

        self.previous_location = current_location

        if self.wrong_location() > 0:
            self.wrong_steps += 1

    def wrong_location(self):
        dif = abs(abs(self.transform.rotation.yaw) - self.map.get_waypoint(self.location).transform.rotation.yaw)

        if 90 <= dif <= 270:
            return 1
        elif self.map.get_waypoint(self.location, project_to_road=False,
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
        if self.obstacle is not None:
            vel = self.obstacle.other_actor.get_velocity()
            other_speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            loc = self.obstacle.other_actor.get_location()
            distance = self.calculate_distance(self.location, loc)
        else:
            distance = 1000
            other_speed = 100

        return [[self.speed, distance, other_speed]]

    def get_KPI(self):
        return self.distance, len(self.collision_hist) > 0, self.wrong_steps, self.previous_distance_to_destination

    def calculate_distance(self, location_a, location_b):
        return math.sqrt((location_a.x - location_b.x) ** 2 + (location_a.y - location_b.y) ** 2)

    def append_obstacle(self, event):
        self.obstacle = event

    def car_following(self, car_following):
        desired_velocity = self.vehicle.get_speed_limit() * 0.95

        if self.obstacle is not None:
            v = self.obstacle.other_actor.get_velocity()
            other_velocity = math.sqrt(v.x ** 2 + v.y ** 2)

            return car_following.get_action(self.speed, self.obstacle.distance, desired_velocity, other_velocity)
        else:
            return car_following.get_action(self.speed, -1, desired_velocity, -1)

    def shield(self, action_list):
        if self.obstacle is not None:
            closest_object_distance = self.calculate_distance(self.location, self.obstacle.other_actor.get_location())

            return shield(action_list, self.speed, closest_object_distance)
        else:
            return action_list[0]
