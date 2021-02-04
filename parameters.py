# Simulation
MODEL_NAME = "initialize_comparison_0"

ENVIRONMENT = 1
SECONDS_PER_EPISODE = 50
EPISODES = 400

RENDERING = True
OTHER_TRAFFIC = True
SHIELD = False
CAR_FOLLOWING = False

SAVE_EXPERIENCES = False
INITIAL_SPEED = 7
AMOUNT_OF_VEHICLES = 100

SIMPLE_REWARD_A = 1
SIMPLE_REWARD_B = 1

FPS = 1
ACTION_TO_STATE_TIME = 1.5

# Shield
BUFFER_TIME = 2
BUFFER_DISTANCE = 5

INITIALIZE_REPLAY_MEMORY = False
INITIALIZE_REPLAY_SIZE = 250

# Q-learning
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 300
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 10

LEARNING_RATE = 0.00025

DISCOUNT = 0.95#0.99

MIN_EPSILON = 0.001
EPSILON_DECAY = MIN_EPSILON**(1/EPISODES)
EPSILON_DECAY_LINEAR = True

ETA = 0.9

# Other
AGGREGATE_STATS_EVERY = 25
