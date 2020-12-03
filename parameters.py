# Simulation
MODEL_NAME = "shield_test_off"

SECONDS_PER_EPISODE = 15
EPISODES = 250

RENDERING = True
OTHER_TRAFFIC = False
SHIELD = True

DESTINATION_DISTANCE = 120
AMOUNT_OF_VEHICLES = 80

# State and action
WIDTH = 150
HEIGHT = 336
PIXELS_PER_METER = 4

FPS = 4
ACTION_TO_STATE_TIME = 0.1

#STEER_ACTIONS = [-1,-0.5,0,0.5,1]
STEER_ACTIONS = [0]
ACC_ACTIONS = [-1,-0.5,0,0.5,1]
STEER_PROBABILITIES = [1]#[0.1, 0.2, 0.4, 0.2, 0.1]
ACC_PROBABILITIES = [0.05, 0.1, 0.15, 0.4, 0.4]

# Shield
BUFFER_TIME = 0.4
BUFFER_DISTANCE = 5

# Q-learning
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 10

LEARNING_RATE = 0.00025

DISCOUNT = 0.99

MIN_EPSILON = 0.01
EPSILON_DECAY = MIN_EPSILON**(1/EPISODES)
SPEED_LIMIT_EXPLORATION = 0.7

# Other
MEMORY_FRACTION = 0.8
AGGREGATE_STATS_EVERY = 10
