import os
from functools import partial
import logging
import re

import torch
from bistiming import Stopwatch
from mkdir_p import mkdir_p
import numpy as np

from .utils import set_random_seed
from spurious_ml.models.torch_model import TorchModel
from spurious_ml.variables import get_file_name
from spurious_ml.datasets import add_spurious_correlation, add_colored_spurious_correlation


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

base_model_dir = './models/mem_inference/'

def get_spurious_X(X, version, seed, flatten_result=True):
    n_channels = X.shape[-1]
    if n_channels == 3:
        add_spu_feat_fn = partial(add_colored_spurious_correlation, version=version, seed=seed)
    else:
        add_spu_feat_fn = partial(add_spurious_correlation, version=version, seed=seed)
    ret = add_spu_feat_fn(np.copy(X))
    if flatten_result:
        ret = ret.reshape(len(ret), -1)
    return ret

def get_modified_data(auto_var):
    ds_name = auto_var.get_variable_name("dataset")
    template = r"(?P<ds>[a-z0-9]+)v(?P<version>[0-9a-z]+)?-(?P<sp_counts>[0-9]+)-(?P<cls_no>[0-9])-(?P<seed>[0-9]+)"
    m = re.fullmatch(template, ds_name)
    spurious_version = 'v' + m['version']
    seed = int(m['seed'])
    if m['ds'].startswith("sp"):
        name = str(m['ds'])[2:]
    else:
        name = str(m['ds'])
    tar_trnX, _, tar_tstX, _, shadow_trnX, _, shadow_tstX, _, _ = auto_var.get_var_with_argument("dataset", name, seed=seed)
    #n_channels = tar_trnX.shape[-1]

    model_name = auto_var.get_variable_name("model")
    flatten_result = ("MLP" in model_name)

    modified_tar_trnX = get_spurious_X(tar_trnX, spurious_version, seed, flatten_result)
    modified_tar_tstX = get_spurious_X(tar_tstX, spurious_version, seed, flatten_result)
    modified_shadow_trnX = get_spurious_X(shadow_trnX, spurious_version, seed, flatten_result)
    modified_shadow_tstX = get_spurious_X(shadow_tstX, spurious_version, seed, flatten_result)

    return modified_tar_trnX, modified_tar_tstX, modified_shadow_trnX, modified_shadow_tstX


def run_mem_inference(auto_var):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _ = set_random_seed(auto_var)

    (tar_trnX, tar_trny, tar_tstX, tar_tsty, shadow_trnX, shadow_trny,
     shadow_tstX, shadow_tsty, spurious_ind) = auto_var.get_var("dataset")
    model_name = auto_var.get_variable_name("model")
    if "MLP" in model_name:
        tar_trnX, tar_tstX = tar_trnX.reshape(len(tar_trnX), -1), tar_tstX.reshape(len(tar_tstX), -1)
        shadow_trnX, shadow_tstX = shadow_trnX.reshape(len(shadow_trnX), -1), shadow_tstX.reshape(len(shadow_tstX), -1)
        is_img_data = False
    else:
        is_img_data = True

    ds_name = auto_var.get_variable_name("dataset")
    if ds_name not in ["memmnist", "memfashion", "memcifar10"]:
        (modified_tar_trnX, modified_tar_tstX, modified_shadow_trnX, modified_shadow_tstX) = get_modified_data(auto_var)

    result = {'spurious_ind': spurious_ind}

    multigpu = False

    mkdir_p(base_model_dir)
    result['target_model_path'] = os.path.join(
            base_model_dir, get_file_name(auto_var) + "_target.pt")
    tar_model = auto_var.get_var("model", trnX=tar_trnX, trny=tar_trny, multigpu=multigpu, device=device)
    tar_model.tst_ds = (tar_tstX, tar_tsty)
    if os.path.exists(result['target_model_path']):
        tar_model.load(result['target_model_path'])
    else:
        with Stopwatch("Fitting Model", logger=logger):
            _ = tar_model.fit(tar_trnX, tar_trny, is_img_data=is_img_data)
        tar_model.save(result['target_model_path'])
    result['target_tar_trn_pred'] = tar_model.predict_real(tar_trnX)
    result['target_tar_tst_pred'] = tar_model.predict_real(tar_tstX)
    result['target_shadow_trn_pred'] = tar_model.predict_real(shadow_trnX)
    result['target_shadow_tst_pred'] = tar_model.predict_real(shadow_tstX)
    result['target_tar_trn_acc'] = (tar_model.predict(tar_trnX) == tar_trny).mean()
    result['target_tar_tst_acc'] = (tar_model.predict(tar_tstX) == tar_tsty).mean()
    result['target_shadow_trn_acc'] = (tar_model.predict(shadow_trnX) == shadow_trny).mean()
    result['target_shadow_tst_acc'] = (tar_model.predict(shadow_tstX) == shadow_tsty).mean()

    if ds_name not in ["memmnist", "memfashion", "memcifar10"]:
        result['target_mod_tar_trn_pred'] = tar_model.predict_real(modified_tar_trnX)
        result['target_mod_tar_tst_pred'] = tar_model.predict_real(modified_tar_tstX)
        result['target_mod_shadow_trn_pred'] = tar_model.predict_real(modified_shadow_trnX)
        result['target_mod_shadow_tst_pred'] = tar_model.predict_real(modified_shadow_tstX)
    else:
        (ttar_trnX, _, ttar_tstX, _, tshadow_trnX, _, tshadow_tstX, _, _) = auto_var.get_var_with_argument("dataset", ds_name)
        flatten_result = ("MLP" in model_name)
        Xs = [ttar_trnX, ttar_tstX, tshadow_trnX, tshadow_tstX]
        result['target_mod_preds'] = {}
        for spu in ['v1', 'v3', 'v8']:
            result['target_mod_preds'][(spu, None)] = []
            for tX in Xs:
                result['target_mod_preds'][(spu, None)].append(
                    tar_model.predict_real(get_spurious_X(tX, spu, None, flatten_result)))
        for spu in ['v18', 'v19', 'v20']:
            for rs in range(10):
                result['target_mod_preds'][(spu, rs)] = []
                for tX in Xs:
                    result['target_mod_preds'][(spu, rs)].append(
                        tar_model.predict_real(get_spurious_X(tX, spu, rs, flatten_result)))

    result['aux_model_path'] = os.path.join(
            base_model_dir, get_file_name(auto_var) + "_aux.pt")
    aux_model = auto_var.get_var("model", trnX=shadow_trnX, trny=shadow_trny, multigpu=multigpu, device=device)
    aux_model.tst_ds = (shadow_tstX, shadow_tsty)
    if os.path.exists(result['aux_model_path']):
        aux_model.load(result['aux_model_path'])
    else:
        with Stopwatch("Fitting Model", logger=logger):
            _ = aux_model.fit(shadow_trnX, shadow_trny, is_img_data=is_img_data)
        aux_model.save(result['aux_model_path'])
    result['aux_tar_trn_pred'] = aux_model.predict_real(tar_trnX)
    result['aux_tar_tst_pred'] = aux_model.predict_real(tar_tstX)
    result['aux_shadow_trn_pred'] = aux_model.predict_real(shadow_trnX)
    result['aux_shadow_tst_pred'] = aux_model.predict_real(shadow_tstX)
    result['aux_tar_trn_acc'] = (aux_model.predict(tar_trnX) == tar_trny).mean()
    result['aux_tar_tst_acc'] = (aux_model.predict(tar_tstX) == tar_tsty).mean()
    result['aux_shadow_trn_acc'] = (aux_model.predict(shadow_trnX) == shadow_trny).mean()
    result['aux_shadow_tst_acc'] = (aux_model.predict(shadow_tstX) == shadow_tsty).mean()

    if ds_name not in ["memmnist", "memfashion", "memcifar10"]:
        result['aux_mod_tar_trn_pred'] = aux_model.predict_real(modified_tar_trnX)
        result['aux_mod_tar_tst_pred'] = aux_model.predict_real(modified_tar_tstX)
        result['aux_mod_shadow_trn_pred'] = aux_model.predict_real(modified_shadow_trnX)
        result['aux_mod_shadow_tst_pred'] = aux_model.predict_real(modified_shadow_tstX)
    else:
        (ttar_trnX, _, ttar_tstX, _, tshadow_trnX, _, tshadow_tstX, _, _) = auto_var.get_var_with_argument("dataset", ds_name)
        flatten_result = ("MLP" in model_name)
        Xs = [ttar_trnX, ttar_tstX, tshadow_trnX, tshadow_tstX]
        result['target_mod_preds'] = {}
        for spu in ['v1', 'v3', 'v8']:
            result['target_mod_preds'][(spu, None)] = []
            for tX in Xs:
                result['target_mod_preds'][(spu, None)].append(
                    tar_model.predict_real(get_spurious_X(tX, spu, None, flatten_result)))
        for spu in ['v18', 'v19', 'v20']:
            for rs in range(10):
                result['target_mod_preds'][(spu, rs)] = []
                for tX in Xs:
                    result['target_mod_preds'][(spu, rs)].append(
                        tar_model.predict_real(get_spurious_X(tX, spu, rs, flatten_result)))

    print(result)
    return result
