import os

import numpy as np
#import tensorflow as tf
import torch

from spurious_ml.variables import get_file_name

def set_random_seed(auto_var):
    random_seed = auto_var.get_var("random_seed")

    torch.manual_seed(random_seed)
    #tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    random_state = np.random.RandomState(auto_var.get_var("random_seed"))
    auto_var.set_intermidiate_variable("random_state", random_state)

    return random_state

def load_model(auto_var, trnX, trny, n_channels, model_dir="./models", device=None):
    model = auto_var.get_var("model", trnX=trnX, trny=trny, n_channels=n_channels, device=device)
    model_path = get_file_name(auto_var).split("-")
    model_path = os.path.join(model_dir, model_path + '.pt')

    model.load(model_path)
    return model_path, model
