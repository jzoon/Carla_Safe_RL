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


class CarEnv3:
    STATE_LENGTH = 10
    STATE_WIDTH = 4

    STEER_ACTIONS = [-1, 1]
    ACC_ACTIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]
    AMOUNT_OF_ACTIONS = 7

    actor_list = []
    collision_hist = []
    distance = 0
    previous_location = None
    speed = 0
    acceleration = 0
    colsensor = None
    episode_start = 0
    state = None
    location = None
    velocity = None
    transform = None
    obstacle = None

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(4.0)

        self.world = self.client.load_world('Town06')
        self.world.set_weather(carla.WeatherParameters.ClearSunset)
        self.map = self.world.get_map()

        if not RENDERING:
            settings = self.world.get_settings()
            settings.no_rendering_mode = True
            self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter('model3')[0]
        self.model_3.set_attribute('color', '255,0,0')

        self.start_transform = self.world.get_map().get_spawn_points()[265]

        self.destination = self.world.get_map().get_spawn_points()[265]
        self.destination.location.x -= 400
        self.destination.location.y += 6

        spawn_npc.main()

        self.shield_object = shield(self.ACC_ACTIONS)

        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(self.start_transform.location + carla.Location(z=50),
                                                carla.Rotation(pitch=-90)))

    def reset(self):
        self.distance = 0
        self.previous_location = None
        self.obstacle = None
        self.wrong_action = False

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
            while self.calculate_distance(self.vehicle.get_location(), self.obstacle.other_actor.get_location()) < 15:
                time.sleep(0.1)

        velocity = carla.Vector3D(-INITIAL_SPEED, 0, 0)
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

        vehicle_list = self.world.get_actors().filter("vehicle.*")

        if len(vehicle_list) < 75:
            spawn_npc.main()

        self.vehicle.set_transform(self.map.get_waypoint(
            self.vehicle.get_transform().location).transform)

        return self.state, reward, done, action

    def update_parameters(self):
        self.location = self.vehicle.get_location()
        self.transform = self.vehicle.get_transform()
        self.velocity = self.vehicle.get_velocity()
        self.speed = math.sqrt(self.velocity.x ** 2 + self.velocity.y ** 2 + self.velocity.z ** 2)
        self.state = self.get_state()

    def get_reward_and_done(self):
        done = False
        reward = self.reward_simple()

        if self.passed_destination(self.location, self.previous_location):
            return reward, True

        if len(self.collision_hist) != 0 or self.wrong_action:
            done = True
            reward = -SIMPLE_REWARD_B

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return reward, done

    def reward_simple(self):
        v_max = self.vehicle.get_speed_limit()
        if v_max < 1:
            v_max = 50

        return SIMPLE_REWARD_A * (self.speed / v_max)

    def car_control(self, action):
        if action < len(self.STEER_ACTIONS):
            waypoint = self.map.get_waypoint(self.location)
            if action == 0:
                if self.waypoint_left(waypoint) and self.lane_change_possible(waypoint, True, 3):
                    self.vehicle.set_transform(waypoint.get_left_lane().transform)
                else:
                    self.wrong_action = True
            else:
                if self.waypoint_right(waypoint) and self.lane_change_possible(waypoint, False, 3):
                    self.vehicle.set_transform(waypoint.get_right_lane().transform)
                else:
                    self.wrong_action = True
        else:
            acc_action = action - len(self.STEER_ACTIONS)
            if self.ACC_ACTIONS[acc_action] < 0:
                self.vehicle.apply_control(
                    carla.VehicleControl(brake=-self.ACC_ACTIONS[acc_action])
                )
            else:
                self.vehicle.apply_control(
                    carla.VehicleControl(throttle=self.ACC_ACTIONS[acc_action])
                )

    def update_KPIs(self, current_location):
        if self.previous_location is None:
            self.previous_location = current_location
        else:
            self.distance += self.calculate_distance(current_location, self.previous_location)

        self.previous_location = current_location

    def passed_destination(self, current_location, previous_location):
        up_x = max(current_location.x, previous_location.x) + 10
        down_x = min(current_location.x, previous_location.x) - 10
        up_y = max(current_location.y, previous_location.y) + 10
        down_y = min(current_location.y, previous_location.y) - 10

        if down_x < self.destination.location.x < up_x and down_y < self.destination.location.y < up_y:
            return True

        return False

    def get_state(self):
        vehicle_list = self.world.get_actors().filter("vehicle.*")

        distances = []
        closest_vehicles = []

        for vehicle in vehicle_list:
            if vehicle != self.vehicle:
                vehicle_location = vehicle.get_location()
                distance = self.calculate_distance(self.location, vehicle_location)

                if len(distances) < self.STATE_LENGTH - 2:
                    distances.append(distance)
                    closest_vehicles.append(vehicle)
                elif distance < max(distances):
                    index = distances.index(max(distances))
                    distances[index] = distance
                    closest_vehicles[index] = vehicle

        waypoint = self.map.get_waypoint(self.location)

        state = [[int(self.waypoint_left(waypoint)), 0, 0, int(self.waypoint_right(waypoint))],
                 [self.location.x, self.location.y, self.velocity.x, self.velocity.y]]

        while len(closest_vehicles) > 0:
            index = distances.index(min(distances))
            location = closest_vehicles[index].get_location()
            velocity = closest_vehicles[index].get_velocity()
            state.append([location.x, location.y, velocity.x, velocity.y])
            del distances[index]
            del closest_vehicles[index]

        return state

    def lane_change_possible(self, waypoint, left, distance):
        if left:
            location = waypoint.get_left_lane().transform.location
        else:
            location = waypoint.get_right_lane().transform.location

        vehicle_list = self.world.get_actors().filter("vehicle.*")

        for vehicle in vehicle_list:
            if abs(vehicle.get_location().x - location.x) < distance:
                return False

        return True

    def get_KPI(self):
        return self.calculate_distance(self.location, self.start_transform.location), len(self.collision_hist) > 0 or self.wrong_action, 0

    def calculate_distance(self, location_a, location_b):
        return math.sqrt((location_a.x - location_b.x) ** 2 + (location_a.y - location_b.y) ** 2)

    def append_obstacle(self, event):
        self.obstacle = event

    def waypoint_right(self, waypoint):
        if waypoint.lane_change == carla.LaneChange.Right or waypoint.lane_change == carla.LaneChange.Both:
            return True

        return False

    def waypoint_left(self, waypoint):
        if waypoint.lane_change == carla.LaneChange.Left or waypoint.lane_change == carla.LaneChange.Both:
            return True

        return False

    def car_following(self, car_following):
        desired_velocity = self.vehicle.get_speed_limit() * 0.95

        if self.obstacle is not None:
            v = self.obstacle.other_actor.get_velocity()
            other_velocity = math.sqrt(v.x ** 2 + v.y ** 2)

            return car_following.get_action(self.speed, self.obstacle.distance, desired_velocity, other_velocity)
        else:
            return car_following.get_action(self.speed, -1, desired_velocity, -1)

    def shield(self, action_list):
        for action in action_list:
            waypoint = self.map.get_waypoint(self.location)
            if action == 0:
                if self.waypoint_left(waypoint) and self.lane_change_possible(waypoint, True, 6):
                    return action
            elif action == 1:
                if self.waypoint_left(waypoint) and self.lane_change_possible(waypoint, True, 6):
                    return action
            else:
                if self.obstacle is None:
                    return action

                closest_object_distance = self.calculate_distance(self.location,
                                                                  self.obstacle.other_actor.get_location())

                if self.shield_object.is_safe(action-2, self.speed, closest_object_distance):
                    return action

        return 2
