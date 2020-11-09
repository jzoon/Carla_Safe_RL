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
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions


class CarEnv:
    actor_list = []
    collision_hist = []
    lane_hist = []

    speed = 0

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(4.0)

        self.world = self.client.load_world('Town01')
        self.world.set_weather(carla.WeatherParameters.ClearSunset)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter('model3')[0]

        self.birdview_producer = BirdViewProducer(
            self.client,
            target_size=PixelDimensions(width=WIDTH, height=HEIGHT),
            pixels_per_meter=PIXELS_PER_METER,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA)

        spawn_npc.main()

    def reset(self):
        self.distance = 0
        self.wrong_steps = 0
        self.previous_location = None

        self.collision_hist = []
        self.actor_list = []
        self.lane_hist = []

        self.transform = self.world.get_map().get_spawn_points()[2]

        self.vehicle = None

        while self.vehicle is None:
            self.vehicle = self.world.try_spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        colsensor = self.blueprint_library.find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_hist.append(event))

        lanesensor = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lanesensor = self.world.spawn_actor(lanesensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.lanesensor)
        self.lanesensor.listen(lambda event: self.lane_hist.append(event))

        self.episode_start = time.time()

        return self.get_state()

    def step(self, action):
        self.car_control(action)

        reward = 0
        done = False

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        self.speed = kmh

        wrong_loc = self.wrong_location()

        if wrong_loc == 1:
            reward -= 10
        elif wrong_loc == 2:
            reward -= 20

        reward += int(kmh/5)

        reward -= len(self.lane_hist)
        self.lane_hist = []

        if len(self.collision_hist) != 0:
            done = True
            reward = -500

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.get_state(), reward, done, None

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

    def update_KPIs(self):
        if self.previous_location == None:
            self.previous_location = self.vehicle.get_transform().location
        else:
            loc = self.vehicle.get_transform().location
            self.distance += math.sqrt((loc.x - self.previous_location.x)**2 + (loc.y - self.previous_location.y)**2)
            self.previous_location = loc

        if self.wrong_location() > 0:
            self.wrong_steps += 1

    def wrong_location(self):
        dif = abs(abs(self.vehicle.get_transform().rotation.yaw) - self.world.get_map().get_waypoint(
            self.vehicle.get_location()).transform.rotation.yaw)

        if 90 <= dif <= 270:
            return 1
        elif self.world.get_map().get_waypoint(self.vehicle.get_location(), project_to_road=False,
                                               lane_type=carla.LaneType.Sidewalk) is not None:
            return 2

        return 0

    def get_state(self):
        return self.birdview_producer.produce(agent_vehicle=self.vehicle).transpose((2, 1, 0))

    def get_KPI(self):
        return self.distance, self.wrong_steps

    def get_speed_limit(self):
        return self.vehicle.get_speed_limit()*3.6
