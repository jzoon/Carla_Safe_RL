# Simulation
MODEL_NAME = "data_generation"

SECONDS_PER_EPISODE = 50
EPISODES = 250

RENDERING = False
OTHER_TRAFFIC = True
SHIELD = False
CAR_FOLLOWING = False

SAVE_EXPERIENCES = False

DESTINATION_DISTANCE = 160
AMOUNT_OF_VEHICLES = 100
INITIAL_SPEED = 7

REWARD_FUNCTION = 'simple' # simple, complex
SIMPLE_REWARD_A = 1
SIMPLE_REWARD_B = 1
DEST = True

STATE_LENGTH = 1
STATE_WIDTH = 3

FPS = 1
ACTION_TO_STATE_TIME = 1.5

#STEER_ACTIONS = [-1,-0.5,0,0.5,1]
STEER_ACTIONS = [0]
ACC_ACTIONS = [-1.0,-0.5,0.0,0.5,1.0]

# Shield
BUFFER_TIME = 2
BUFFER_DISTANCE = 5

# Q-learning
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 10

LEARNING_RATE = 0.00025

DISCOUNT = 0.95#0.99

MIN_EPSILON = 0.001
EPSILON_DECAY = MIN_EPSILON**(1/EPISODES)
SPEED_LIMIT_EXPLORATION = 0.3
EPSILON_DECAY_LINEAR = True

ETA = 0.9

# Other
MEMORY_FRACTION = 0.8
AGGREGATE_STATS_EVERY = 25
