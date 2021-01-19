#!/usr/bin/env python
import rospy
import numpy as np
import rosbag
from std_msgs.msg import Int32
from std_msgs.msg import String
from std_msgs.msg import Float64
import pickle
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from carla_msgs.msg import CarlaEgoVehicleStatus
from carla_msgs.msg import CarlaEgoVehicleControl
from automotive_control_msgs.msg import LongitudinalActionGoal
from gazebo_msgs.msg import ModelStates
import carla
import glob
import random
import time
import math

secure_random = random.SystemRandom()

class CarlaLongitudinalControl(object):

    def __init__(self):

        self.control_loop_rate = rospy.Rate(10)  # 10Hz
        self.host = rospy.get_param('/carla/host', '127.0.0.1')
        self.port = rospy.get_param('/carla/port', 2000)
        self.timeout = rospy.get_param('/carla/timeout', 10)
        self.accel_target = 0.0
        self.vel_target = 0.0
        self.acc_current = 0.0
        self.velocity = 0.0
        self.gear = 0
        self.throttle_set = 0.0
        self.throttle_current = 0.0
        self.throttleVals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                             -0.92, -0.93, -0.94, -0.95, -0.96, -0.97, -0.98,
                             -0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0]
        self.openLoopData = []
        self.acc_goal = []
        self.vel_goal = []
        self.acc_bag_time = []
        self.vel_bag_time = []
        self.count = 0
        self.gain = 0.1
        self.acc_feedback = 0.0
        self.accgoalcopynf = 0.0

        # ego

        self.player_ego_id = None

        # hero
        self.world = None
        self.actor_filter = None
        self.actor_spawnpoint = carla.Transform(
            carla.Location(x=3, y=-0.2, z=1),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
        self.role_name = None
        self.player = None
        self.player_ego = None
        self.hero_vel_x = []
        self.hero_vel_y = []
        self.hero_pos_x = []
        self.hero_pos_y = []
        self.hero_time = []
        self.count_hero = 0
        self.player_id = None

        ###

        self.controlMsg = CarlaEgoVehicleControl()
        # self.controlheroMsg = CarlaEgoVehicleControl()

        self.Status = rospy.Subscriber("/carla/ego_vehicle/vehicle_status",
                                            CarlaEgoVehicleStatus, self.VehicleStatus)

        # self.SetpointSub = rospy.Subscriber("/profia1/vehicle/longitudinal_action/goal",
        #                                     LongitudinalActionGoal, self.setpointCallback)

        # self.controlPub = rospy.Publisher("/carla/ego_vehicle/vehicle_control_cmd", CarlaEgoVehicleControl,
        #                                   queue_size=1)
        # self.controlheroPub = rospy.Publisher("/carla/hero/vehicle_control_cmd", CarlaEgoVehicleControl,
        #                                   queue_size=1)

        # self.controlaccGoal = LongitudinalActionGoal()
        # self.controlvelGoal = LongitudinalActionGoal()
        # self.goalaccPub = rospy.Publisher("/carla/ego_vehicle/vehicle_goal_acc", LongitudinalActionGoal, queue_size=1)
        # self.goalvelPub = rospy.Publisher("/carla/ego_vehicle/vehicle_goal_vel", LongitudinalActionGoal, queue_size=1)
        self.egoGoal = CarlaEgoVehicleStatus()
        self.egoGoalPub = rospy.Publisher("carla/ego_vehicle/egoGoal", CarlaEgoVehicleStatus, queue_size=1)

        self.hero_twist = Twist()
        self.hero_pose = Pose()
        self.heroGoal = ModelStates()
        self.heroGoal.twist.append(self.hero_twist)
        self.heroGoal.pose.append(self.hero_pose)
        self.heroGoalPub = rospy.Publisher("carla/hero/heroGoal", ModelStates, queue_size=1)

        self.controlMsg.gear = 1
        # self.controlPub.publish(self.controlMsg)

    def VehicleStatus(self, data):
        self.velocity = data.velocity
        self.gear = data.control.gear
        self.acc_current = data.acceleration.linear.x
        self.throttle_current = data.control.throttle

    # def setpointCallback(self, data):
    #     # rospy.loginfo(rospy.get_caller_id() + " Setpoint %s \n", data.goal.setpoint)
    #     self.accel_target = data.goal.setpoint

    def spawn_ego(self):
        client = carla.Client(self.host, self.port)
        client.set_timeout(self.timeout)
        self.world = client.get_world()
        self.player_ego_id = self.world.get_actors().filter("vehicle.tesla.model3")
        self.player_ego = self.world.get_actor(self.player_ego_id[0].id)

        # self.player_ego.set_target_velocity(carla.Vector3D(x=3.0,
        #                                                y=0.0,
        #                                                z=0.0))

        # self.player_ego.set_location(carla.Location(x=-297.2, y=245.1, z=1))
        self.player_ego.set_transform(carla.Transform(carla.Location(x=267.5, y=151.0, z=1),
                                                      carla.Rotation(pitch=0.0, yaw=0.2141, roll=0.0)))
        time.sleep(3)


        # self.role_name = "ego_vehicle"
        # self.actor_spawnpoint = carla.Transform(carla.Location(x=-190.5, y=247.2, z=1),
        #                                         carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
        #
        # # Get vehicle blueprint.
        # self.actor_filter = "vehicle.tesla.model3"
        # blueprint = secure_random.choice(
        #     self.world.get_blueprint_library().filter(self.actor_filter))
        # blueprint.set_attribute('role_name', "{}".format(self.role_name))
        # if blueprint.has_attribute('color'):
        #     color = secure_random.choice(blueprint.get_attribute('color').recommended_values)
        #     blueprint.set_attribute('color', color)
        # # Spawn the vehicle.
        # self.player_ego = self.world.try_spawn_actor(blueprint, self.actor_spawnpoint)
        # print("Ego vehicle spawned")
        # self.world = client.get_world()
        # print(self.world.get_actors())
        # # time.sleep(3)

    def spawn_hero(self):

        # from bp
        client = carla.Client(self.host, self.port)
        client.set_timeout(self.timeout)
        self.world = client.get_world()
        self.role_name = "hero"
        self.actor_spawnpoint = carla.Transform(carla.Location(x=187.5, y=147.5, z=1),
                                                carla.Rotation(pitch=0.0, yaw=0.2141, roll=0.0))

        # Get vehicle blueprint.
        self.actor_filter = "vehicle.dodge_charger.police"
        blueprint = secure_random.choice(
            self.world.get_blueprint_library().filter(self.actor_filter))
        blueprint.set_attribute('role_name', "{}".format(self.role_name))
        if blueprint.has_attribute('color'):
            color = secure_random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the vehicle.
        self.player = self.world.try_spawn_actor(blueprint, self.actor_spawnpoint)
        print("Hero vehicle spawned")
        time.sleep(3)

    def hero_control(self):
        theta = 0.2141 * math.pi / 180
        self.player.set_target_velocity(carla.Vector3D(
            x=self.hero_vel_x[self.count_hero] * math.cos(theta) + -self.hero_vel_y[self.count_hero] * math.sin(theta),
            y=-self.hero_vel_y[self.count_hero] * math.cos(theta) + self.hero_vel_x[self.count_hero] * math.sin(theta),
            z=0.0))
        self.heroGoal.twist[0].linear.x = self.hero_vel_x[self.count_hero]
        self.heroGoal.twist[0].linear.y = self.hero_vel_y[self.count_hero]
        self.heroGoal.pose[0].position.x = self.hero_pos_x[self.count_hero]
        self.heroGoal.pose[0].position.y = self.hero_pos_y[self.count_hero]
        self.heroGoalPub.publish(self.heroGoal)
        # self.controlheroPub.publish(self.controlheroMsg)
        if self.hero_vel_x[self.count_hero] > 13:
            self.player.set_light_state(carla.VehicleLightState.Special1)

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def run_accel_control_loop(self):

        self.accel_target = self.acc_goal[self.count]
        self.vel_target = self.vel_goal[self.count_hero]
        # self.vel_target = self.vel_goal[self.find_nearest(self.vel_bag_time, self.acc_bag_time[self.count])]
        if -15.48 > self.accel_target > 15.48 or -1 > self.vel_target > 27:
            print("error acceleration or velocity target out of range")
            exit(1)
        i = 0
        throttle_lower = 0.0
        throttle_upper = 1.0
        lower_acc_diff = 15.48  # abs(self.accel_target)  # 15.48
        upper_acc_diff = 15.48  # - abs(self.accel_target)  # 15.48

        # feedback based on velocity error
        # error_vel = self.vel_target - self.velocity
        # self.acc_feedback = self.gain * error_vel
        self.accgoalcopynf = self.accel_target
        # self.accel_target = self.accel_target + self.acc_feedback

        # #####

        for b in self.openLoopData:

            if self.vel_target <= 1 and self.accel_target >= 0.0:  # 0.31 for acc and 0.95 for brake
                throttle_lower = 0.2
                throttle_upper = 0.2
                i = i + 1
                continue
            elif self.vel_target <= 1 and self.accel_target < 0.0:
                throttle_lower = -0.2
                throttle_upper = -0.2
                i = i + 1
                continue
            else:
                indx = self.find_nearest(b[0], self.vel_target)
            if abs(b[0][indx] - self.velocity) > 1:
                i = i + 1
                continue
            # print((b[0])[indx], (b[1])[indx])
            b_acc_diff = (b[1])[indx] - self.accel_target
            # print("acc diff:", b_acc_diff)

            if b_acc_diff < 0.0 and abs(b_acc_diff) < abs(lower_acc_diff):
                lower_acc_diff = abs(b_acc_diff)
                throttle_lower = b[2][indx]
                # print("throttle lower, vel ", b_acc_diff, throttle_lower, b[0][indx])
            elif b_acc_diff >= 0 and abs(b_acc_diff) < abs(upper_acc_diff):
                upper_acc_diff = abs(b_acc_diff)
                throttle_upper = b[2][indx]

            i = i + 1

        #print(throttle_lower, lower_acc_diff, upper_acc_diff)
        prop_acc = (lower_acc_diff / (abs(lower_acc_diff) + abs(upper_acc_diff))) * (throttle_upper - throttle_lower)
        throttle_required = throttle_lower + prop_acc

        # initial condition - no throttle!
        if self.accel_target == 0 and self.vel_target == 0:
            throttle_required = 0.0

        if throttle_required < 0.0:
            self.controlMsg.brake = throttle_required
            self.controlMsg.throttle = 0.0
            self.player_ego.apply_control(carla.VehicleControl(throttle=0.0, brake=abs(throttle_required)))
            self.player_ego.set_light_state(carla.VehicleLightState.Brake)
        else:
            self.controlMsg.brake = 0.0
            self.controlMsg.throttle = throttle_required
            self.player_ego.apply_control(carla.VehicleControl(throttle=throttle_required, brake=0.0))
            self.player_ego.set_light_state(carla.VehicleLightState.NONE)
        # self.controlPub.publish(self.controlMsg)

        # publish goal data
        # self.controlaccGoal.goal.setpoint = self.accgoalcopynf
        # self.controlvelGoal.goal.setpoint = self.vel_target
        # self.goalaccPub.publish(self.controlaccGoal)
        # self.goalvelPub.publish(self.controlvelGoal)

        self.egoGoal.acceleration.linear.x = self.accgoalcopynf
        self.egoGoal.velocity = self.vel_target
        self.egoGoalPub.publish(self.egoGoal)

        print("acc goal:", self.accel_target)
        print("acc curr:", self.acc_current)
        # print("throttle low:", throttle_lower, "throttle up:", throttle_upper)
        # print("throttle req:", throttle_required, "prop acc:", prop_acc)
        # print("lad:", lower_acc_diff, "uad:", upper_acc_diff)
        print("throttle", self.controlMsg.throttle, "brake:", self.controlMsg.brake)
        print("vel targ:", self.vel_target)
        print("vel curr:", self.velocity)
        # self.velocity += self.accel_target * 0.1

    def destroy(self):
        if self.player:
            self.player.destroy()
            self.player = None
            print("Hero vehicle destroyed")
        # if self.player_ego:
        #     self.player_ego.destroy()
        #     self.player_ego = None
        #     print("Ego vehicle destroyed")

    def run(self):
        """

        Control loop

        :return:
        """
        # spawn vehicles
        self.spawn_ego()
        self.spawn_hero()

        # load 2D control map
        for i in self.throttleVals:
            stringSave = "/media/data/PBN/OpenLoopTest/Tesla/ThrottleSetData1/AccVelCurve/data_" + str(i) + ".txt"
            with open(stringSave, "rb") as fp:
                b = pickle.load(fp)
                self.openLoopData.append(b)

        # load goal rosbag
        bag = rosbag.Bag('/home/dev/test_data/TULIP_UC15_AI_T41.xosc_2020-11-27-10-56-42.bag')

        for topic, msg, t in bag.read_messages(topics=['/profia1/vehicle/longitudinal_action/goal']):
            self.acc_goal.append(msg.goal.setpoint)
            self.acc_bag_time.append(t.to_sec())

        for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states/']):
            if len(msg.twist) > 2:
                self.vel_bag_time.append(t.to_sec())
                self.vel_goal.append(msg.twist[1].linear.x)
                self.hero_vel_x.append(msg.twist[2].linear.x)
                self.hero_vel_y.append(msg.twist[2].linear.y)
                self.hero_pos_x.append(msg.pose[2].position.x)
                self.hero_pos_y.append(msg.pose[2].position.y)
                self.hero_time.append(t.to_sec())
            elif len(msg.twist) > 1:
                self.vel_bag_time.append(t.to_sec())
                self.vel_goal.append(msg.twist[1].linear.x)
                self.hero_vel_x.append(0.0)
                self.hero_vel_y.append(0.0)
                self.hero_pos_x.append(0.0)
                self.hero_pos_y.append(0.0)
                self.hero_time.append(t.to_sec())
            elif len(msg.twist) > 0:
                self.vel_bag_time.append(t.to_sec())
                self.vel_goal.append(0.0)
                self.hero_vel_x.append(0.0)
                self.hero_vel_y.append(0.0)
                self.hero_pos_x.append(0.0)
                self.hero_pos_y.append(0.0)
                self.hero_time.append(t.to_sec())

        bag.close()

        self.count_hero = self.find_nearest(self.vel_bag_time, self.acc_bag_time[0])

        # control loop
        while not rospy.is_shutdown():
            self.hero_control()
            self.run_accel_control_loop()
            if (self.count + 10) < len(self.acc_goal):
                self.count = self.count + 10
                self.count_hero = self.count_hero + 10
            print(rospy.get_rostime().to_sec())

            try:
                self.control_loop_rate.sleep()
            except rospy.ROSInterruptException:
                pass




def main():
    """

    main function

    :return:
    """
    rospy.init_node('rosPubAckermann', anonymous=True)
    controller = CarlaLongitudinalControl()
    try:
        controller.run()
    finally:
        # del controller
        controller.destroy()  # pbn add
        rospy.loginfo("Done")


if __name__ == "__main__":
    main()
