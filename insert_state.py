from collections import deque
from keras.models import load_model
import numpy as np
from parameters import *

MODEL_PATH = "models/SIP_shield_test_more_actions_____4.27max____2.91avg___-1.00min-1614313374.model"

if __name__ == "__main__":

    model = load_model(MODEL_PATH)
    state = np.expand_dims([[5, 100, 10]], axis=0)
    print(model.predict(state))
