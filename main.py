from parameters import *
import random
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from DQNAgent import *
from CarEnv2 import *
from CarFollowing import *
import os
from threading import Thread
from tqdm import tqdm


if __name__ == '__main__':
    epsilon = 1
    ep_rewards = []
    colissions = []
    distances = []
    wrong_locations = []
    times = []

    save_episodes = []
    save_rewards = []
    save_distances = []
    save_times = []
    save_collisions = []

    if SAVE_EXPERIENCES:
        file_name = r"experiences/" + MODEL_NAME + "_" + str(int(time.time())) + ".csv"
        f = open(file_name, "x")
        f.close()

    #random.seed(1)
    #np.random.seed(1)
    #tf.set_random_seed(1)

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    #backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    if ENVIRONMENT == 1:
        env = CarEnv1()
    elif ENVIRONMENT == 2:
        env = CarEnv2()
    else:
        env = CarEnv3()

    agent = DQNAgent(env.STATE_LENGTH, env.STATE_WIDTH, len(env.ACC_ACTIONS)*len(env.STEER_ACTIONS))
    car_follow = CarFollowing(env.ACC_ACTIONS)

    if INITIALIZE_REPLAY_MEMORY:
        env.shield_object.initialize_replay_memory(INITIALIZE_REPLAY_SIZE, agent)

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)
    agent.get_qs(np.ones((1, env.STATE_LENGTH, env.STATE_WIDTH)))

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        env.collision_hist = []
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()
        previous_time = time.time()

        while not done:
            current_time = time.time()

            if np.random.random() > epsilon:
                action_list = np.argsort(agent.get_qs(np.expand_dims(current_state, axis=0)))[::-1]
            else:
                action_list = list(range(len(env.ACC_ACTIONS) * len(env.STEER_ACTIONS)))
                random.shuffle(action_list)

                if CAR_FOLLOWING:
                    if np.random.random() < ETA:
                        follow_action = env.car_following(car_follow)
                        index = action_list.index(follow_action)
                        action_list[index] = action_list[0]
                        action_list[0] = follow_action

            time_spent = time.time() - previous_time
            if time_spent < 1/FPS:
                time.sleep(1/FPS - time_spent)
                previous_time = time.time()

            new_state, reward, done, chosen_action = env.step(action_list)

            episode_reward += reward
            agent.update_replay_memory((current_state, chosen_action, reward, new_state, done))

            if chosen_action != action_list[0]:
                agent.update_replay_memory((current_state, action_list[0], -SIMPLE_REWARD_B, new_state, True))

            if SAVE_EXPERIENCES:
                f = open(file_name, "a")
                f.write(str(current_state) + "," + str(chosen_action) + "," + str(reward) + "," + str(new_state) + "," + str(done) + "\n")
                f.close()

            current_state = new_state
            step += 1

        for actor in env.actor_list:
            actor.destroy()

        distances.append(max(1, env.get_KPI()[0]))
        colissions.append(int(env.get_KPI()[1]))
        wrong_locations.append(env.get_KPI()[2]/step)
        times.append(time.time() - episode_start)

        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            if sum(distances[-AGGREGATE_STATS_EVERY:]) > 0:
                avg_colissions_per_m = sum(colissions[-AGGREGATE_STATS_EVERY:])/sum(distances[-AGGREGATE_STATS_EVERY:])
            else:
                avg_colissions_per_m = sum(colissions[-AGGREGATE_STATS_EVERY:])
            avg_wrong_location = sum(wrong_locations[-AGGREGATE_STATS_EVERY:]) / len(wrong_locations[-AGGREGATE_STATS_EVERY:])
            avg_speed = sum(distances[-AGGREGATE_STATS_EVERY:])/sum(times[-AGGREGATE_STATS_EVERY:])

            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon, collisions_per_km=avg_colissions_per_m*1000,
                                           wrong_location=avg_wrong_location, speed=avg_speed)

            save_episodes.append(episode)
            save_rewards.append(average_reward)
            save_distances.append(sum(distances[-AGGREGATE_STATS_EVERY:])/len(distances[-AGGREGATE_STATS_EVERY:]))
            save_times.append(sum(times[-AGGREGATE_STATS_EVERY:])/len(times[-AGGREGATE_STATS_EVERY:]))
            save_collisions.append(sum(colissions[-AGGREGATE_STATS_EVERY:])/len(colissions[-AGGREGATE_STATS_EVERY:]))

        if epsilon > MIN_EPSILON:
            if EPSILON_DECAY_LINEAR:
                epsilon -= 1/(0.9*EPISODES)
            else:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

    all_data = np.array([save_episodes, save_distances, save_times, save_collisions, save_rewards]).transpose()
    np.savetxt(r"manual_logs/" + MODEL_NAME + "_" + str(int(time.time())) + ".csv", all_data, delimiter=",", header="episode,distance,time,collision,reward", comments="")

    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
