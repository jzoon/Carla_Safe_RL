from parameters import *
import spawn_npc
import math
import time
import sys
import glob
import os
import matplotlib.pyplot as plt
from PredictNewStateModel import predict_new_state

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
        self.map = self.world.get_map()

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

        self.start_transform.location.x -= 50
        asfd = self.world.spawn_actor(self.model_3, self.start_transform)
        self.start_transform.location.x += 50
        self.actor_list.append(asfd)

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
        self.state = self.get_state()
        self.location = self.vehicle.get_location()
        self.transform = self.vehicle.get_transform()

        return self.state

    def step(self, action_list):
        self.state = self.get_state()
        self.transform = self.vehicle.get_transform()
        self.location = self.transform.location
        v = self.vehicle.get_velocity()
        self.speed = int(math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        a = self.vehicle.get_acceleration()
        self.acceleration = int(math.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2))

        if SHIELD:
            action = self.shield(action_list)
        else:
            action = action_list[0]

        #w_location = self.vehicle.get_location()
        #w_acc = -self.vehicle.get_acceleration().x
        #w_speed = -self.vehicle.get_velocity().x

        #p_distance, p_speed, p_acc = predict_new_state(w_speed, w_acc, action, 2)
        #temp_time = time.time()

        self.car_control(action)
        time.sleep(ACTION_TO_STATE_TIME)

        self.state = self.get_state()
        self.transform = self.vehicle.get_transform()
        self.location = self.transform.location
        v = self.vehicle.get_velocity()
        self.speed = int(math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        self.update_KPIs(self.location)
        reward, done = self.get_reward_and_done(action, action_list)

        #time.sleep(2 - (time.time() - temp_time))
        #new_loc = self.vehicle.get_location()
        #driven = self.calculate_distance(new_loc.x, w_location.x, new_loc.y, w_location.y)
        #distance_dif = p_distance - driven
        #new_vel = -self.vehicle.get_velocity().x
        #speed_dif = p_speed - new_vel
        #new_acc = -self.vehicle.get_acceleration().x
        #acc_dif = p_acc - new_acc

        #with open("plots/" + str(action) + ".csv", "a") as file:
            #write_str = str(distance_dif) + ',' + str(speed_dif) + "," + str(acc_dif) + '\n'
            #file.write(write_str)

        return self.state, reward, done, None

    def get_reward_and_done(self, action, action_list):
        reward = 0
        done = False

        if action != action_list[0]:
            reward -= 50

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
            self.distance += self.calculate_distance(current_location.x, self.previous_location.x, current_location.y, self.previous_location.y)

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
            else:
                print(str(action) + " is not safe!")

        print("No safe action found. RIP")
        return 0

    def is_safe(self, action):
        distance, new_speed, _ = predict_new_state(self.speed, self.acceleration, action, BUFFER_TIME)
        new_x = (WIDTH/2)
        new_y = (HEIGHT/2) - int(distance*PIXELS_PER_METER)

        return self.check_safe_trajectory(int(new_x), int(new_y), new_speed, 0)

    def check_safe_trajectory(self, x, y, new_speed, angle):
        distance = self.get_safe_distance_blocks(new_speed)

        x_angle = math.sin(math.radians(angle))
        y_angle = math.cos(math.radians(angle))

        current_distance = 0

        #if abs(x - WIDTH/2) < 8 and abs(y - HEIGHT/2) < 8:
        #    current_distance = 8

        rgb = BirdViewProducer.as_rgb(self.state.transpose([2, 1, 0]))

        while current_distance < distance:
            block1 = self.dangerous_block(x + math.floor(x_angle*current_distance), y - math.floor(y_angle*current_distance))
            block2 = self.dangerous_block(x + math.floor(x_angle*current_distance), y - math.ceil(y_angle*current_distance))
            block3 = self.dangerous_block(x + math.ceil(x_angle * current_distance), y - math.floor(y_angle * current_distance))
            block4 = self.dangerous_block(x + math.ceil(x_angle * current_distance), y - math.ceil(y_angle * current_distance))

            rgb[y - math.floor(y_angle * current_distance), x + math.floor(x_angle * current_distance)] = [0, 0, 0]
            rgb[y - math.ceil(y_angle * current_distance), x + math.floor(x_angle * current_distance)] = [0, 0, 0]
            rgb[y - math.floor(y_angle * current_distance), x + math.ceil(x_angle * current_distance)] = [0, 0, 0]
            rgb[y - math.ceil(y_angle * current_distance), x + math.ceil(x_angle * current_distance)] = [0, 0, 0]

            if block1 or block2 or block3 or block4:
                plt.imshow(rgb)
                plt.show()
                return False

            current_distance += 1

        return True

    def dangerous_block(self, x, y):
        if x < 0 or x > WIDTH or y < 0 or y > HEIGHT:
            #print("Out of state")
            return False

        if self.state[x, y, 4] == 1:
            #print("Safe: own car")
            return False
        elif self.state[x, y, 0] == 0:
            #print("Unsafe: no road")
            return True
        elif self.state[x, y, 3] == 1:
            #print("Unsafe: car")
            return True
        elif self.state[x, y, 8] == 1:
            #print("Unsafe: pedestrian")
            return True
        else:
            #print("Safe")
            return False

    def get_safe_distance_blocks(self, speed):
        meters = 0

        while speed > 0.1:
            dist, speed, _ = predict_new_state(speed, 0, 0, 0.1)
            meters += dist

        return int(meters*PIXELS_PER_METER + BUFFER_DISTANCE*PIXELS_PER_METER)
