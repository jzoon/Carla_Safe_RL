from parameters import *
import random
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from DQNAgent import *
from CarEnv import *
import os
from threading import Thread
from tqdm import tqdm


def pick_random_action():
    steering_probabilities = [0.1, 0.2, 0.4, 0.2, 0.1]
    acc_probabilities = [0.05, 0.1, 0.15, 0.4, 0.4]
    steer_draw = random.choices(range(len(STEER_ACTIONS)), weights=steering_probabilities)[0]
    acc_draw = random.choices(range(len(ACC_ACTIONS)), weights=acc_probabilities)[0]

    return 5 * steer_draw + acc_draw


if __name__ == '__main__':
    epsilon = 1
    ep_rewards = []
    times = []
    distances = []
    wrong_locations = []

    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    agent = DQNAgent()
    env = CarEnv()

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)
    agent.get_qs(np.ones((1, WIDTH, HEIGHT, 9)))

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        env.collision_hist = []
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()

        while not done:
            current_time = time.time()

            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(np.expand_dims(current_state, axis=0)))
            else:
                if env.speed < 0.7*env.get_speed_limit():
                    action = pick_random_action()
                else:
                    action = random.choice(range(len(ACC_ACTIONS)*len(STEER_ACTIONS)))

            time_spent = time.time() - current_time
            if time_spent < 1/FPS:
                time.sleep(1/FPS - time_spent)

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            current_state = new_state
            step += 1

        for actor in env.actor_list:
            actor.destroy()

        times.append(time.time()-episode_start)
        distances.append(env.get_KPI()[0])
        wrong_locations.append(env.get_KPI()[1]/step)

        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            avg_time = sum(times[-AGGREGATE_STATS_EVERY:]) / len(times[-AGGREGATE_STATS_EVERY:])
            avg_distance = sum(distances[-AGGREGATE_STATS_EVERY:]) / len(distances[-AGGREGATE_STATS_EVERY:])
            avg_wrong_location = sum(wrong_locations[-AGGREGATE_STATS_EVERY:]) / len(wrong_locations[-AGGREGATE_STATS_EVERY:])

            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon, sim_time=avg_time, distance_traveled=avg_distance, wrong_location=avg_wrong_location)

            # Save model, but only when min reward is greater or equal a set value
            #if episode % int(EPISODES/10) == 0:
            #    agent.model.save(
            #        f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

