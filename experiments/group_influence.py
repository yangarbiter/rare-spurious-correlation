import os
import logging
import re

import torch
from mkdir_p import mkdir_p
import numpy as np

from .utils import set_random_seed
from spurious_ml.datasets import add_spurious_correlation
from spurious_ml.variables import get_file_name
from spurious_ml.influence_utils.influence import first_order_group_influence, calc_influence_single


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

base_model_dir = './models/train_classifier/'
save_model_dir = './models/group_influence/'

def run_group_influence(auto_var):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _ = set_random_seed(auto_var)
    trnX, trny, tstX, tsty, spurious_ind = auto_var.get_var("dataset")
    model_name = auto_var.get_variable_name("model")
    if "MLP" in model_name:
        trnX, tstX = trnX.reshape(len(trnX), -1), tstX.reshape(len(tstX), -1)
        is_img_data = False
    else:
        is_img_data = True

    model_path = os.path.join(base_model_dir, auto_var.get_var('model_path'))

    result = {
        'spurious_ind': spurious_ind,
        'model_path': os.path.join(save_model_dir, get_file_name(auto_var) + ".pt")
    }

    multigpu = False
    model = auto_var.get_var("model", trnX=trnX, trny=trny, multigpu=multigpu, device=device)
    model.tst_ds = (tstX, tsty)

    model.load(model_path)

    model.model.eval()

    ds_name = auto_var.get_variable_name("dataset")
    if "mnist" in ds_name:
        template = r"mnist(?P<version>v[0-9a-z]+)?-(?P<sp_counts>[0-9]+)-(?P<cls_no>[0-9])-(?P<seed>[0-9]+)"
        groupdict = re.fullmatch(template, ds_name).groupdict()
        print(groupdict)
        cls_no = int(groupdict['cls_no'])
        seed = int(groupdict['seed'])
        version = groupdict['version']

        for j in range(len(tsty)):
            if trny[j] == cls_no:
                result['influences_same_cls_idx'] = j
                same_cls_tstX = add_spurious_correlation(tstX[j:j+1].reshape(1, 28, 28), version=version, seed=seed)
                same_yy = tsty[j:j+1]
                break
        for j in range(len(tsty)):
            if trny[j] !=  cls_no:
                result['influences_dif_cls_idx'] = j
                dif_cls_tstX = add_spurious_correlation(tstX[j:j+1].reshape(1, 28, 28), version=version, seed=seed)
                dif_yy = tsty[j:j+1]
                break

        influences = np.array(calc_influence_single(
            model.model,
            torch.from_numpy(same_cls_tstX.reshape(1, -1)).float(),
            torch.from_numpy(same_yy).long(),
            torch.from_numpy(trnX).float(),
            torch.from_numpy(trny).long(),
            recursion_depth=2000,
            r=10,
        ))
        result['influences_same_cls'] = influences
        influences = np.array(calc_influence_single(
            model.model,
            torch.from_numpy(dif_cls_tstX.reshape(1, -1)).float(),
            torch.from_numpy(dif_yy).long(),
            torch.from_numpy(trnX).float(),
            torch.from_numpy(trny).long(),
            recursion_depth=2000,
            r=10,
        ))
        result['influences_dif_cls'] = influences

    scale = auto_var.get_var("gi_scale")
    recursion_depth = auto_var.get_var("gi_depth")
    if "fashion" in ds_name:
        scale = 100 if scale == float("-inf") else scale
        damp = 0.01
        if recursion_depth == 987654321:
            if len(spurious_ind) >= 100:
                recursion_depth = 1000
            else:
                recursion_depth = 2000
    elif "cifar10" in ds_name:
        scale = 100 if scale == float("-inf") else scale
        damp = 0.01
        trnX = trnX.transpose(0, 3, 1, 2)
        if recursion_depth == 987654321:
            recursion_depth = 1000
    else:
        scale = 25 if scale == float("-inf") else scale
        damp = 0.01
        if recursion_depth == 987654321:
            if len(spurious_ind) >= 100:
                recursion_depth = 1000
            else:
                recursion_depth = 2000

    params = first_order_group_influence(
        model.model,
        torch.from_numpy(trnX[spurious_ind]).float(),
        torch.from_numpy(trny[spurious_ind]).long(),
        torch.from_numpy(trnX).float(),
        torch.from_numpy(trny).long(),
        recursion_depth=recursion_depth,
        r=10,
        scale=scale,
        damp=damp,
    )

    with torch.no_grad():
        i = 0
        for param in model.model.parameters():
            if param.requires_grad:
                param.copy_(param - params[i])
                i += 1
    model.save(result['model_path'])

    if "cifar10" in ds_name:
        trnX = trnX.transpose(0, 2, 3, 1)

    result['trn_acc'] = (model.predict(trnX) == trny).mean()
    result['tst_acc'] = (model.predict(tstX) == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")
    return result
