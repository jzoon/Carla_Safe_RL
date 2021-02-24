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

TEST_POLICY = False
RANDOM = False

MODEL_PATHS = ["models/test_no_carla.model"]
EPISODES = 50

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

                    print(qs)

                new_state, reward, done, _ = env.step(action)
                episode_reward += reward
                current_state = new_state

                if done:
                    break

                frame_time = time.time() - step_start
                fps_counter.append(frame_time)
                #print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: {action} | Reward: {reward}')

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

