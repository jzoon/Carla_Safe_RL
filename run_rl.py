from collections import deque
import numpy as np
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from parameters import *
from CarEnv1 import *
from CarEnv2 import *
from CarEnv3 import *
from tqdm import tqdm
import matplotlib.pyplot as plt

PLOT_BEHAVIOR = False
TEST_POLICY = True
RANDOM = False

MODEL_PATHS = ["models/super_safe_policy_shield_1000_____6.23max____3.48avg____1.79min-1614211774.model"]
EPISODES = 100

if __name__ == "__main__":
    env = CarEnv2()

    all_average_rewards = []
    all_average_collisions = []
    all_average_speeds = []

    for MODEL_PATH in MODEL_PATHS:
        model = load_model(MODEL_PATH)
        fps_counter = deque(maxlen=15)
        model.predict(np.ones((1, env.STATE_LENGTH, env.STATE_WIDTH)))

        all_rewards = []
        all_collisions = []
        all_distances = []
        all_times = []

        for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
            current_state = env.reset()
            env.collision_hist = []
            step = 0
            start_time = time.time()
            episode_reward = 0

            own_speed = []
            other_speed = []
            obstacle_distance = []

            while True:
                step += 1
                step_start = time.time()

                if TEST_POLICY:
                    action = [env.car_following()]
                elif RANDOM:
                    action = [random.randint(0, 4)]
                else:
                    qs = model.predict(np.expand_dims(current_state, axis=0))[0]
                    action = np.argsort(qs)[::-1]

                    #print(qs)

                new_state, reward, done, _ = env.step(action)
                episode_reward += reward
                current_state = new_state

                if done:
                    break

                frame_time = time.time() - step_start
                fps_counter.append(frame_time)
                #print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: {action} | Reward: {reward}')

                if PLOT_BEHAVIOR:
                    own_speed.append(env.speed)
                    obstacle_vel = env.obstacle.other_actor.get_velocity()
                    other_speed.append(math.sqrt(obstacle_vel.x ** 2 + obstacle_vel.y ** 2 + obstacle_vel.z ** 2))
                    obstacle_distance.append(env.calculate_distance(env.location, env.obstacle.other_actor.get_location()))

            all_times.append(time.time() - start_time)
            distance, col = env.get_KPI()
            all_distances.append(distance)
            if step > 1:
                all_collisions.append(col)
            else:
                all_collisions.append(0)
            all_rewards.append(episode_reward)

            for actor in env.actor_list:
                actor.destroy()

            if PLOT_BEHAVIOR:
                ax1 = plt.plot(list(range(step-1)), own_speed, label="Speed AV (m/s)")
                ax2 = plt.plot(list(range(step-1)), other_speed, label="Speed front vehicle (m/s)")
                ax3 = plt.plot(list(range(step - 1)), obstacle_distance, label="Distance (m)")
                plt.legend()
                plt.savefig("tempplots/" + str(episode))
                plt.show()

        print()
        print(MODEL_PATH)
        print("Average reward: " + str(np.mean(all_rewards)))
        print("Collisions per km: " + str(sum(all_collisions)/(sum(all_distances)/1000)))
        print("Average speed: " + str(sum(all_distances)/sum(all_times)))
        print()

        all_average_rewards.append(np.mean(all_rewards))
        all_average_collisions.append(sum(all_collisions)/(sum(all_distances)/1000))
        all_average_speeds.append(sum(all_distances)/sum(all_times))

    print("Average reward: Mean: " + str(np.mean(all_average_rewards)) + ", Std: " + str(np.std(all_average_rewards)))
    print("Collisions per km: Mean: " + str(np.mean(all_average_collisions)) + ", Std: " + str(np.std(all_average_collisions)))
    print("Average speed: Mean: " + str(np.mean(all_average_speeds)) + ", Std: " + str(np.std(all_average_speeds)))

