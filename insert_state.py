from collections import deque
from keras.models import load_model
import numpy as np
from parameters import *

MODEL_PATH = "models/save_done_____3.81max____2.72avg____0.56min__1611141184.model"

if __name__ == "__main__":

    model = load_model(MODEL_PATH)
    state = np.expand_dims([[5, 5, 0]], axis=0)
    print(model.predict(state))
