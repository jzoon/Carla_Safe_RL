from collections import deque
from keras.models import load_model
from GymEnv import *


MODEL_PATH = "gym_models/first_gym_test____16.40max___15.37avg___14.73min__1610631013.model"

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    env = GymEnv()
    fps_counter = deque(maxlen=15)
    model.predict(np.ones((1, STATE_NUMBER_OF_VEHICLES, STATE_WIDTH)))

    for i in range(20):
        print("Restarting episode")

        current_state = env.reset()
        env.collision_hist = []
        step = 0
        start_time = time.time()

        while True:
            step += 1
            step_start = time.time()
            qs = model.predict(np.expand_dims(current_state, axis=0))[0]
            action = np.argsort(qs)[::-1]

            new_state, reward, done, _ = env.step(action[0])
            current_state = new_state

            if done:
                break

            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: {action} | Reward: {reward}')
            print(qs)

            print()
