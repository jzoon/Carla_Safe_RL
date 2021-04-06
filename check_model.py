from keras.models import load_model
import numpy as np
from DQNAgentLoss import *

MODEL_PATH = "models/Scenario2_Shield1_SIPshield0-test_new_loss__1617259823.model"

if __name__ == "__main__":
    model = load_model(MODEL_PATH, custom_objects={'shield_loss': shield_loss})
    state = np.expand_dims([[8, 20, 5, 0,0,0,0,0,0,0,0,0,0,0]], axis=0)
    print(model.predict(state))
