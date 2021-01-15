from parameters import *
import random
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from DQNAgent import *
from CarEnv import *
from GymEnv import *
from CarFollowing import *
import os
from threading import Thread
from tqdm import tqdm


if __name__ == '__main__':
    epsilon = 1
    ep_rewards = []
    colissions = []
    speeds = []
    distances = []

    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    agent = DQNAgent()
    env = GymEnv()

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)
    agent.get_qs(np.ones((1, STATE_NUMBER_OF_VEHICLES, STATE_WIDTH)))

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()
        previous_time = time.time()
        info = None

        while not done:
            current_time = time.time()

            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(np.expand_dims(current_state, axis=0)))
            else:
                action = random.choice(list(range(len(ACC_ACTIONS) * len(STEER_ACTIONS))))

            time_spent = time.time() - previous_time

            new_state, reward, done, info = env.step(action)
            episode_reward += reward

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            current_state = new_state

            step += 1

        if info['crashed']:
            colissions.append(1)
        else:
            colissions.append(0)
        speeds.append(env.get_KPI()[0])
        distances.append(env.get_KPI()[1])

        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            if sum(distances[-AGGREGATE_STATS_EVERY:]) > 0:
                avg_colissions_per_m = sum(colissions[-AGGREGATE_STATS_EVERY:])/sum(distances[-AGGREGATE_STATS_EVERY:])
            else:
                avg_colissions_per_m = sum(colissions[-AGGREGATE_STATS_EVERY:])
            avg_speed = sum(speeds[-AGGREGATE_STATS_EVERY:])/len(speeds[-AGGREGATE_STATS_EVERY:])

            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon, collisions_per_km=avg_colissions_per_m*1000,
                                           speed=avg_speed)

            # Save model, but only when min reward is greater or equal a set value
            #if episode % int(EPISODES/10) == 0:
            #    agent.model.save(
            #        f'temp_models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        if epsilon > MIN_EPSILON:
            if EPSILON_DECAY_LINEAR:
                epsilon -= 1/EPISODES
            else:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'gym_models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
