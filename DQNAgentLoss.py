from parameters import *
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import layers
import keras.backend as kb
from collections import deque
import tensorflow as tf
import time
import random
from shield import shield


# This alternative loss function punishes the agent if it generates high Q-values for actions that would be overruled
# by a shield.
def shield_loss(y_true, y_predict):
    loss = kb.mean(kb.square(y_true[:, :11] - y_predict[:, :11]))
    terms = (kb.exp(y_predict[:, :11])) / (kb.sum(kb.exp(y_predict[:, :11])) + 0.001)
    loss += LAMBDA * (kb.sum(y_true[:, 11:] * terms))

    return loss

# This class defines the DDQN agent methods when an alternative loss function is used as the shield-based learning
# method.
class DQNAgent:
    # Initializes the neural network.
    def __init__(self, state_length, state_width, output_size, model_time):
        self.state_width = state_width + output_size
        self.output_size = output_size * 2
        self.state_length = state_length

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(model_time)}")

        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

        self.shield = shield([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Creates the neural network model, with an input layer, output layer and three hidden layers. Parallel to the NN,
    # an extra input is forwarded through the model which indicates whether actions would be overruled or not.
    def create_model(self):
        inputs = layers.Input(shape=(self.state_length, self.state_width,))
        input_state = layers.Lambda(lambda x: x[:, :, :3])(inputs)
        input_p = layers.Lambda(lambda x: x[:,:, 3:])(inputs)

        layer1 = layers.Dense(32, activation="relu")(input_state)
        layer2 = layers.Dense(64, activation="relu")(layer1)
        layer3 = layers.Flatten()(layer2)
        action = layers.Dense(int(self.output_size/2), activation="linear")(layer3)

        flat_p = layers.Flatten()(input_p)
        concat = layers.Concatenate()([action, flat_p])

        model = Model(inputs=inputs, outputs=concat)
        model.compile(loss=shield_loss, optimizer=Adam(lr=LEARNING_RATE, clipnorm=1.0), metrics=["accuracy"])

        return model

    # Adds an experience to the replay memory.
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains the DDQN. If the replay buffer is large enough, a minibatch is sampled on which the DDQN is trained by
    # using the Bellman equation and the target network. The overruled actions are appended in this process so that
    # they can be used by the alternative loss function.
    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch], dtype=object)

        shield_list = []
        for state in current_states:
            action_list = []

            for action in range(11):
                if self.shield.is_safe(action, state[0, 0], state[0, 1]):
                    action_list.append(0.0)
                else:
                    action_list.append(1.0)

            shield_list.append(action_list)

        shield_list = np.expand_dims(shield_list, 1)
        current_states = np.concatenate((current_states, shield_list), axis=2)

        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])
        new_current_states = np.concatenate((new_current_states, shield_list), axis=2)

        with self.graph.as_default():
            future_qs_list_target = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        with self.graph.as_default():
            future_qs_list_online = self.model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q_arg = np.argmax(future_qs_list_online[index, :3])
                max_future_q = future_qs_list_target[index, max_future_q_arg]
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            current_state = np.concatenate((current_state, shield_list[index]), axis=1)

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step
            self.target_update_counter += 1

        with self.graph.as_default():
            self.model.fit(np.array(X), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

    # Updates the target network.
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    # Returns the Q-values for a state.
    def get_qs(self, state):
        return self.model.predict(np.concatenate((state, np.zeros([state.shape[0], 1, 11])), 2))[0, :11]

    # Is used to be able to train the network in a loop parallel to the agent exploring the environment.
    def train_in_loop(self):
        X = np.random.uniform(size=(1, self.state_length, self.state_width)).astype(np.float32)
        y = np.random.uniform(size=(1, self.output_size)).astype(np.float32)

        with self.graph.as_default():
            self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return

            self.train()
            time.sleep(0.01)

# This class initializes the TensorBoard, which can be used to visualize the process.
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
