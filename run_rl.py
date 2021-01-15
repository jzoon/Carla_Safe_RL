from collections import deque
import numpy as np
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from parameters import *
from CarEnv import *


MODEL_PATH = "models/bug_fix_test___290.92max__192.64avg__138.54min__1610640882.model"

if __name__ == "__main__":
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    #backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    model = load_model(MODEL_PATH)
    env = CarEnv()
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

            new_state, reward, done, _ = env.step(action)
            current_state = new_state

            if done:
                break

            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: {action} | Reward: {reward}')

            print()

        print()
        print("Colissions per m: " + str(int(env.get_KPI()[1]/env.get_KPI()[0])))
        print("Distance to destination: " + str(env.get_KPI()[3]))
        print("Percentage wrong steps: " + str(env.get_KPI()[2]/step))
        print()

        for actor in env.actor_list:
            actor.destroy()
