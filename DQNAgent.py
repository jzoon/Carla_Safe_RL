from parameters import *
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import layers
from collections import deque
import tensorflow as tf
import time
import random

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        inputs = layers.Input(shape=(STATE_NUMBER_OF_VEHICLES, 5,))

        layer1 = layers.Dense(64, activation="relu")(inputs)
        layer2 = layers.Dense(128, activation="relu")(layer1)
        layer3 = layers.Dense(256, activation="relu")(layer2)

        layer4 = layers.Flatten()(layer3)

        action = layers.Dense(len(STEER_ACTIONS)*len(ACC_ACTIONS), activation="linear")(layer4)

        model = Model(inputs=inputs, outputs=action)
        optimizer = Adam(lr=LEARNING_RATE, clipnorm=1.0)
        model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch], dtype=object)

        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])

        with self.graph.as_default():
            future_qs_list_target = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        with self.graph.as_default():
            future_qs_list_online = self.model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q_arg = np.argmax(future_qs_list_online[index])
                max_future_q = future_qs_list_target[index, max_future_q_arg]
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        with self.graph.as_default():
            self.model.fit(np.array(X), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):

        return self.model.predict(state)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, STATE_NUMBER_OF_VEHICLES, 5)).astype(np.float32)
        y = np.random.uniform(size=(1, len(STEER_ACTIONS) * len(ACC_ACTIONS))).astype(np.float32)

        with self.graph.as_default():
            self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
