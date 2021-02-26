from collections import deque
from keras.models import load_model
import numpy as np
from parameters import *

MODEL_PATH = "models/test_new_SIP_shield_____5.79max____3.49avg____1.56min-1614337333.model"

if __name__ == "__main__":

    model = load_model(MODEL_PATH)
    state = np.expand_dims([[0, 50, 10]], axis=0)
    print(model.predict(state))
