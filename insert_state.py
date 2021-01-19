from collections import deque
from keras.models import load_model
import numpy as np
from parameters import *

MODEL_PATH = "models/long_interval_test_____4.48max____1.36avg___-0.97min__1611066311.model"

if __name__ == "__main__":

    model = load_model(MODEL_PATH)
    state = np.expand_dims([[2, 100, 8]], axis=0)
    print(model.predict(state))
