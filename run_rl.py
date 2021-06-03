from collections import deque
import numpy as np
from keras.models import load_model
from CarEnv1 import *
from CarEnv2 import *
from old.CarEnv3 import *
from tqdm import tqdm
import matplotlib.pyplot as plt

PLOT_BEHAVIOR = False
TEST_POLICY = False
RANDOM = False

MODEL_PATHS = [["models/Scenario2/DDQN/Scenario2_Shield0_SIPshield0-__1618654364-800.model"]]
#MODEL_PATHS = [["models/Scenario2/SCS/Scenario2_Shield1_SIPshield0-__1617192844.model"]]
#MODEL_PATHS = [["models/Scenario2/SIPS/Scenario2_Shield0_SIPshield1-__1618297434-900.model"]]
EPISODES = 100

if __name__ == "__main__":
    all_average_rewards = []
    all_average_collisions = []
    all_average_speeds = []

    for MODEL_GROUP in MODEL_PATHS:
        min_col = 1000
        best_reward = 0
        best_speed = 0
        mp = ""

        for MODEL_PATH in MODEL_GROUP:
            env = CarEnv2()
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
                        action = [random.randint(0, len(env.ACC_ACTIONS) - 1)]
                    else:
                        qs = model.predict(np.expand_dims(current_state, axis=0))[0]
                        action = np.argsort(qs)[::-1]

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
                    print(own_speed)
                    print(other_speed)
                    print(obstacle_distance)
                    ax1 = plt.plot(list(range(step-1)), own_speed, label="Speed AV (m/s)")
                    #ax2 = plt.plot(list(range(step-1)), other_speed, label="Speed front vehicle (m/s)")
                    #ax3 = plt.plot(list(range(step - 1)), obstacle_distance, label="Distance (m)")
                    plt.legend()
                    plt.xlabel("Step")
                    #plt.savefig("tempplots/" + str(episode))
                    #plt.show()

            average_reward = np.mean(all_rewards)
            std_reward = np.std(all_rewards)
            average_col = sum(all_collisions)/(sum(all_distances)/1000)
            std_col = (np.std(np.array(all_collisions)/np.array(all_distances)/1000))
            average_speed = sum(all_distances)/sum(all_times)
            std_speed = (np.std(np.array(all_distances)/np.array(all_times)))

            print("ERWIN")
            print(average_reward)
            print(std_reward)
            print(average_col)
            print(std_col)
            print(average_speed)
            print(std_speed)
            print()

            if average_col < min_col:
                min_col = average_col
                best_reward = average_reward
                best_speed = average_speed
                mp = MODEL_PATH
            elif average_col == min_col and average_speed > best_speed:
                min_col = average_col
                best_reward = average_reward
                best_speed = average_speed
                mp = MODEL_PATH

        print()
        print(mp)
        print("Average reward: " + str(best_reward))
        print("Collisions per km: " + str(min_col))
        print("Average speed: " + str(best_speed))
        print()

        all_average_rewards.append(best_reward)
        all_average_collisions.append(min_col)
        all_average_speeds.append(best_speed)

    print("Average reward: Mean: " + str(np.mean(all_average_rewards)) + ", Std: " + str(np.std(all_average_rewards)))
    print("Collisions per km: Mean: " + str(np.mean(all_average_collisions)) + ", Std: " + str(np.std(all_average_collisions)))
    print("Average speed: Mean: " + str(np.mean(all_average_speeds)) + ", Std: " + str(np.std(all_average_speeds)))

