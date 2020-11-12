from collections import deque
import numpy as np
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from parameters import *
from CarEnv import *


MODEL_PATH = "models/no_traffic_lr_0.00025___265.72max___34.36avg_-207.15min__1605114349.model"

if __name__ == "__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    model = load_model(MODEL_PATH)
    env = CarEnv()
    fps_counter = deque(maxlen=15)
    model.predict(np.ones((1, WIDTH, HEIGHT, 9)))

    for i in range(10):
        print("Restarting episode")

        current_state = env.reset()
        env.collision_hist = []

        start_time = time.time()

        while True:
            step_start = time.time()
            qs = model.predict(np.expand_dims(current_state, axis=0))[0]
            action = np.argmax(qs)

            new_state, reward, done, _ = env.step(action)
            current_state = new_state

            if done:
                break

            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: {action} | Reward: {reward} | qs : {qs}')

        print()
        print("Simulation time: " + str(time.time() - start_time))
        print("Distance driven: " + str(env.get_KPI()[0]))
        print("Percentage wrong steps: " + str(env.get_KPI()[1]))
        print()

        for actor in env.actor_list:
            actor.destroy()
