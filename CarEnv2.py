from parameters import *
import spawn_npc
import math
import time
import sys
import glob
import os
from CarFollowing import *
from shield import shield

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class CarEnv2:
    STATE_LENGTH = 1
    STATE_WIDTH = 3

    ACC_ACTIONS = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    AMOUNT_OF_ACTIONS = len(ACC_ACTIONS)

    actor_list = []
    collision_hist = []
    previous_location = None
    speed = 0
    colsensor = None
    episode_start = 0
    state = None
    location = None
    velocity = None
    transform = None
    obstacle = None
    collision = False

    def __init__(self):
        self.client = carla.Client('localhost', PORT)
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
        self.start_transform.location.x += 28
        self.destination_distance = 200

        if TEST_SCENARIO == 1:
            self.other_transform = self.world.get_map().get_spawn_points()[2]
            self.other_transform.location.x -= 122
            self.world.try_spawn_actor(self.model_3, self.other_transform)
        elif TEST_SCENARIO == 2:
            self.other_transform = self.world.get_map().get_spawn_points()[2]
            self.other_transform.location.x -= 2
            self.other_vehicle = self.world.try_spawn_actor(self.model_3, self.other_transform)
            velocity = carla.Vector3D(-6, 0, 0)
            self.other_vehicle.set_velocity(velocity)
        elif TEST_SCENARIO == 3:
            self.other_transform = self.world.get_map().get_spawn_points()[2]
            self.other_transform.location.x -= 2
            self.other_vehicle = self.world.try_spawn_actor(self.model_3, self.other_transform)
            velocity = carla.Vector3D(-12, 0, 0)
            self.other_vehicle.set_velocity(velocity)
        else:
            spawn_npc.main()

        self.shield_object = shield(self.ACC_ACTIONS)
        self.car_following_object = CarFollowing(self.ACC_ACTIONS)
        self.vel_to_acc = VelToAcc(self.ACC_ACTIONS)

        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(self.start_transform.location + carla.Location(z=50),
                                                carla.Rotation(pitch=-90)))

    def reset(self):
        self.states = []

        self.collision = False
        self.previous_location = None
        self.obstacle = None
        self.collision_hist = []
        self.actor_list = []
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

        obstacle_sensor = self.blueprint_library.find('sensor.other.obstacle')
        obstacle_sensor.set_attribute('only_dynamics', 'True')
        obstacle_sensor.set_attribute('distance', '300')
        self.obstacle_sensor = self.world.spawn_actor(obstacle_sensor, sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.obstacle_sensor)
        self.obstacle_sensor.listen(lambda event: self.append_obstacle(event))

        time.sleep(0.5)
        if self.obstacle is not None:
            while self.calculate_distance(self.vehicle.get_location(), self.obstacle.other_actor.get_location()) < 25:
                time.sleep(0.1)

        velocity = carla.Vector3D(-INITIAL_SPEED, 0, 0)
        self.vehicle.set_velocity(velocity)
        self.episode_start = time.time()

        self.update_parameters()
        self.previous_location = self.location

        return self.state

    def step(self, action_list):
        if TEST_SCENARIO == 2:
            velocity = carla.Vector3D(-6, 0, 0)
            self.other_vehicle.set_velocity(velocity)
        if TEST_SCENARIO == 3:
            if time.time() - self.episode_start > 10:
                self.other_vehicle.apply_control(
                    carla.VehicleControl(brake=-1.0)
                )
            else:
                velocity = carla.Vector3D(-12, 0, 0)
                self.other_vehicle.set_velocity(velocity)

        self.update_parameters()
        self.states.append(self.state)

        if SHIELD:
            action = self.shield(action_list)
        elif SIP_SHIELD:
            action = self.sip_shield(action_list)
        else:
            action = action_list[0]

        self.car_control(action)

        time.sleep(ACTION_TO_STATE_TIME)

        self.update_parameters()

        if self.transform.rotation.yaw != self.map.get_waypoint(self.location).transform.rotation.yaw:
            self.vehicle.set_transform(self.map.get_waypoint(self.location).transform)

        reward, done = self.get_reward_and_done()
        self.previous_location = self.location

        return self.state, reward, done, action

    def update_parameters(self):
        self.location = self.vehicle.get_location()
        self.transform = self.vehicle.get_transform()
        self.velocity = self.vehicle.get_velocity()
        self.speed = math.sqrt(self.velocity.x ** 2 + self.velocity.y ** 2 + self.velocity.z ** 2)
        self.state = self.get_state()

    def get_reward_and_done(self):
        reward = self.reward_simple()

        if self.passed_destination():
            return SIMPLE_REWARD_C, True

        if len(self.collision_hist) != 0:
            self.collision = True
            print(self.states)

            return -SIMPLE_REWARD_B, True

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            return reward, True

        return reward, False

    def reward_simple(self):
        v_max = self.vehicle.get_speed_limit()
        if v_max < 1:
            v_max = 50

        return SIMPLE_REWARD_A * (self.speed / v_max)

    def car_control(self, action):
        if self.ACC_ACTIONS[action] < 0:
            self.vehicle.apply_control(
                carla.VehicleControl(brake=-self.ACC_ACTIONS[action])
            )
        else:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=self.ACC_ACTIONS[action])
            )

    def passed_destination(self):
        if self.calculate_distance(self.location, self.start_transform.location) > self.destination_distance:
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
        return self.calculate_distance(self.location, self.start_transform.location), self.collision

    def calculate_distance(self, location_a, location_b):
        return math.sqrt((location_a.x - location_b.x) ** 2 + (location_a.y - location_b.y) ** 2)

    def append_obstacle(self, event):
        self.obstacle = event

    def car_following(self):
        desired_velocity = self.vehicle.get_speed_limit() * 0.95

        if self.obstacle is not None:
            v = self.obstacle.other_actor.get_velocity()
            other_velocity = math.sqrt(v.x ** 2 + v.y ** 2)

            return self.car_following_object.get_action(self.speed, self.obstacle.distance, desired_velocity, other_velocity)
        else:
            return self.car_following_object.get_action(self.speed, -1, desired_velocity, -1)

    def shield(self, action_list):
        if self.obstacle is not None:
            closest_object_distance = self.calculate_distance(self.location, self.obstacle.other_actor.get_location())

            return self.shield_object.shield(action_list, self.speed, closest_object_distance)
        else:
            return action_list[0]

    def sip_shield(self, action_list):
        if self.obstacle is None:
            return action_list[0]

        sip_action = self.car_following()
        closest_object_distance = self.calculate_distance(self.location, self.obstacle.other_actor.get_location())
        rho = self.get_sip_limits(closest_object_distance)

        if closest_object_distance < BUFFER_DISTANCE:
            return min(action_list[0], sip_action)

        for action in action_list:
            if action <= sip_action + rho:
                return action

        return action_list[0]

    def get_sip_limits(self, closest_object_distance):
        x_min = ((0.5 * self.speed ** 2) / -self.vel_to_acc.get_acc(0, 1)) + BUFFER_DISTANCE
        new_speed = self.vel_to_acc.get_speed(len(self.ACC_ACTIONS) - 1, self.speed, 2)
        brake_distance = (0.5 * new_speed ** 2) / -self.vel_to_acc.get_acc(0, new_speed)
        x_max = brake_distance + self.vel_to_acc.get_distance(len(self.ACC_ACTIONS) - 1, self.speed, 2) + BUFFER_DISTANCE

        if closest_object_distance < x_min:
            return 0
        elif closest_object_distance > x_max:
            return len(self.ACC_ACTIONS)
        else:
            return int(len(self.ACC_ACTIONS) * ((closest_object_distance-x_min) / (x_max-x_min)))
