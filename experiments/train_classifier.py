import os
import logging

import torch
from bistiming import Stopwatch
from mkdir_p import mkdir_p
import numpy as np

from .utils import set_random_seed
from spurious_ml.variables import get_file_name


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

base_model_dir = './models/train_classifier/'

def run_train_classifier(auto_var):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _ = set_random_seed(auto_var)
    trnX, trny, tstX, tsty, spurious_ind = auto_var.get_var("dataset")
    model_name = auto_var.get_variable_name("model")
    if "MLP" in model_name:
        trnX, tstX = trnX.reshape(len(trnX), -1), tstX.reshape(len(tstX), -1)
        is_img_data = False
    else:
        is_img_data = True

    result = {'spurious_ind': spurious_ind}

    multigpu = False
    #multigpu = True if torch.cuda.device_count() > 1 else False

    mkdir_p(base_model_dir)
    result['model_path'] = os.path.join(
            base_model_dir, get_file_name(auto_var) + ".pt")

    model = auto_var.get_var("model", trnX=trnX, trny=trny, multigpu=multigpu, device=device)
    model.tst_ds = (tstX, tsty)

    with Stopwatch("Fitting Model", logger=logger):
        history = model.fit(trnX, trny, is_img_data=is_img_data)
    model.save(result['model_path'])
    result['history'] = history

    result['trn_acc'] = (model.predict(trnX) == trny).mean()
    result['tst_acc'] = (model.predict(tstX) == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")
    return result
