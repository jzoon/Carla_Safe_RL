from CarEnv1 import *
from CarEnv2 import *
from old.CarEnv3 import *
from CarFollowing import *
from threading import Thread
from tqdm import tqdm

if ALT_LOSS:
    from DQNAgentLoss import *
else:
    from DQNAgent import *


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

    if ENVIRONMENT == 1:
        env = CarEnv1()
    elif ENVIRONMENT == 2:
        env = CarEnv2()
    else:
        env = CarEnv3()

    model_time = time.time()
    agent = DQNAgent(env.STATE_LENGTH, env.STATE_WIDTH, env.AMOUNT_OF_ACTIONS, model_time)

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)
    agent.get_qs(np.ones((1, env.STATE_LENGTH, env.STATE_WIDTH)))

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        if episode % 100 == 0 and ENVIRONMENT == 2:
            created = False
            while not created:
                try:
                    env = CarEnv2()
                    created = True
                except Exception as e:
                    continue
            time.sleep(5)
        if episode % UPDATE_TARGET_EVERY == 0:
            agent.update_target_network()

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

            new_state, reward, done, chosen_action = env.step(action_list)

            update_done = done
            if ENVIRONMENT == 1 and reward == 1:
                update_done = False

            episode_reward += reward
            agent.update_replay_memory((current_state, chosen_action, reward, new_state, update_done))

            if chosen_action != action_list[0]:
                if FABRICATE_ACTIONS:
                    agent.update_replay_memory((current_state, action_list[0], -SIMPLE_REWARD_B, current_state, True))
                shield_overrules_episode += 1

            current_state = new_state
            step += 1

        for actor in env.actor_list:
            actor.destroy()

        distances.append(max(1, env.get_KPI()[0]))
        if ENVIRONMENT == 2 and step < 3:
            collisions.append(0)
        else:
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

        if epsilon > 0:
            epsilon -= 1/(EXPLORATION_STOP*EPISODES)
            epsilon = max(0, epsilon)

        if episode == 800:
            agent.model.save(f'models/{MODEL_NAME}__{int(model_time)}-800.model')
        elif episode == 900:
            agent.model.save(f'models/{MODEL_NAME}__{int(model_time)}-900.model')

    all_data = np.array([save_episodes, save_distances, save_times, save_collisions, save_rewards, save_overrules]).transpose()
    np.savetxt(r"manual_logs/" + MODEL_NAME + "-" + str(int(model_time)) + ".csv", all_data, delimiter=",", header="episode,distance,time,collision,return,overrule", comments="")

    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{int(model_time)}.model')
