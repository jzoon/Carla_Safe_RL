from parameters import *
import random
import numpy as np
from DQNAgent import *
from CarEnv1 import *
from CarEnv2 import *
from CarEnv3 import *
from CarFollowing import *
from threading import Thread
from tqdm import tqdm


if __name__ == '__main__':
    epsilon = 1
    ep_rewards = []
    collisions = []
    distances = []
    times = []
    shield_overrules = []
    steps = []

    save_episodes = []
    save_rewards = []
    save_distances = []
    save_times = []
    save_collisions = []
    save_overrules = []

    if SAVE_EXPERIENCES:
        file_name = r"experiences/" + MODEL_NAME + "_" + str(int(time.time())) + ".csv"
        f = open(file_name, "x")
        f.close()

    if ENVIRONMENT == 1:
        env = CarEnv1()
    elif ENVIRONMENT == 2:
        env = CarEnv2()
    else:
        env = CarEnv3()

    agent = DQNAgent(env.STATE_LENGTH, env.STATE_WIDTH, env.AMOUNT_OF_ACTIONS)
    car_follow = CarFollowing(env.ACC_ACTIONS)

    if INITIALIZE_REPLAY_MEMORY:
        env.shield_object.initialize_replay_memory(INITIALIZE_REPLAY_SIZE, agent, env.ACC_ACTIONS, env.STEER_ACTIONS, ENVIRONMENT)

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)
    agent.get_qs(np.ones((1, env.STATE_LENGTH, env.STATE_WIDTH)))

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()
        shield_overrules_episode = 0

        while not done:
            if np.random.random() > epsilon:
                action_list = np.argsort(agent.get_qs(np.expand_dims(current_state, axis=0)))[::-1]
            else:
                action_list = list(range(env.AMOUNT_OF_ACTIONS))
                random.shuffle(action_list)

                if CAR_FOLLOWING:
                    if np.random.random() < ETA:
                        follow_action = env.car_following(car_follow)
                        index = action_list.index(follow_action)
                        action_list[index] = action_list[0]
                        action_list[0] = follow_action

            new_state, reward, done, chosen_action = env.step(action_list)

            episode_reward += reward
            agent.update_replay_memory((current_state, chosen_action, reward, new_state, done))

            if chosen_action != action_list[0]:
                agent.update_replay_memory((current_state, action_list[0], -SIMPLE_REWARD_B, new_state, True))
                shield_overrules_episode += 1

            if SAVE_EXPERIENCES:
                f = open(file_name, "a")
                f.write(str(current_state) + "," + str(chosen_action) + "," + str(reward) + "," + str(new_state) + "," + str(done) + "\n")
                f.close()

            current_state = new_state
            step += 1

        for actor in env.actor_list:
            actor.destroy()

        distances.append(max(1, env.get_KPI()[0]))
        collisions.append(int(env.get_KPI()[1]))
        times.append(time.time() - episode_start)
        ep_rewards.append(episode_reward)
        shield_overrules.append(shield_overrules_episode)
        steps.append(step)

        if not episode % AGGREGATE_STATS_EVERY:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            avg_collisions_per_m = sum(collisions[-AGGREGATE_STATS_EVERY:])/sum(distances[-AGGREGATE_STATS_EVERY:])
            avg_speed = sum(distances[-AGGREGATE_STATS_EVERY:])/sum(times[-AGGREGATE_STATS_EVERY:])
            average_overrule = sum(shield_overrules[-AGGREGATE_STATS_EVERY:]) / sum(steps[-AGGREGATE_STATS_EVERY:])

            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon, collisions_per_km=avg_collisions_per_m*1000,
                                           speed=avg_speed, shield_overrule_percentage=average_overrule)

            save_episodes.append(episode)
            save_rewards.append(average_reward)
            save_distances.append(sum(distances[-AGGREGATE_STATS_EVERY:])/len(distances[-AGGREGATE_STATS_EVERY:]))
            save_times.append(sum(times[-AGGREGATE_STATS_EVERY:])/len(times[-AGGREGATE_STATS_EVERY:]))
            save_collisions.append(sum(collisions[-AGGREGATE_STATS_EVERY:])/len(collisions[-AGGREGATE_STATS_EVERY:]))
            save_overrules.append(average_overrule)

        if epsilon > MIN_EPSILON:
            if EPSILON_DECAY_LINEAR:
                epsilon -= 1/(EXPLORATION_STOP*EPISODES)
            else:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

    all_data = np.array([save_episodes, save_distances, save_times, save_collisions, save_rewards, save_overrules]).transpose()
    np.savetxt(r"manual_logs/" + MODEL_NAME + "-" + str(int(time.time())) + ".csv", all_data, delimiter=",", header="episode,distance,time,collision,reward,overrule", comments="")

    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min-{int(time.time())}.model')
