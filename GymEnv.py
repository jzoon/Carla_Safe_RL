import gym
import highway_env
from matplotlib import pyplot as plt
from parameters import *
import time
import numpy as np

class GymEnv:
    def __init__(self):
        self.env = gym.make('highway-v0')
        self.env.config["lanes_count"] = 2
        self.env.config["action"]["type"] = "ContinuousAction"
        self.env.config["initial_lane_id"] = 1
        self.episode_start = time.time()
        self.speeds = []
        self.episode_end = time.time()
        print(self.env.config["vehicles_count"])

    def reset(self):
        self.episode_start = time.time()
        self.env.reset()
        obs, reward, done, info = self.env.step([0, 0])
        self.speeds = [25]

        return obs_to_state(obs)

    def step(self, action):
        a = int_to_action(action)

        if self.speeds[-1] < 1 and a < 0:
            a = 0

        obs, reward, done, info = self.env.step(a)
        self.env.render()
        self.speeds.append(info['speed'])

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        if done:
            self.episode_end = time.time()

        return obs_to_state(obs), reward, done, info

    def get_KPI(self):
        distance = np.mean(self.speeds) * (self.episode_end - self.episode_start)

        return np.mean(self.speeds), distance


def obs_to_state(obs):
    if len(obs) > STATE_NUMBER_OF_VEHICLES:
        return obs[:STATE_NUMBER_OF_VEHICLES]
    else:
        while len(obs) < STATE_NUMBER_OF_VEHICLES:
            obs.append([0.0,0.0,0.0,0.0,0.0])

    return obs

def int_to_action(action):
    steer_action = int(action / len(ACC_ACTIONS))
    acc_action = action % len(ACC_ACTIONS)

    return [ACC_ACTIONS[acc_action], STEER_ACTIONS[steer_action]]
