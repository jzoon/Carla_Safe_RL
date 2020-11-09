SECONDS_PER_EPISODE = 50

WIDTH = 150
HEIGHT = 336
PIXELS_PER_METER = 4

FPS = 5

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Q300"

LEARNING_RATE = 0.0025

MEMORY_FRACTION = 0.8
MIN_REWARD = 0

EPISODES = 10

DISCOUNT = 0.995
EPSILON_DECAY = 0.999 ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 5

STEER_ACTIONS = [-1,-0.5,0,0.5,1]
ACC_ACTIONS = [-1,-0.5,0,0.5,1]