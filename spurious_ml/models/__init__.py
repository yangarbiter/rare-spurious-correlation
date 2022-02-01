import os

from autovar.base import RegisteringChoiceType, VariableClass, register_var
import numpy as np


DEBUG = int(os.getenv("DEBUG", 0))

class ModelVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Model Variable Class"""
    var_name = "model"

    @register_var(argument=r'(?P<dataaug>[a-zA-Z0-9]+-)?(?P<loss>[a-zA-Z0-9]+)-tor-(?P<arch>[a-zA-Z0-9_]+)(?P<hyper>-[a-zA-Z0-9\.]+)?')
    @staticmethod
    def torch_model_v2(auto_var, inter_var, dataaug, loss, arch, hyper, trnX, trny, device, multigpu=False, **kwargs):
        from .torch_model import TorchModel

        dataaug = dataaug[:-1] if dataaug else None

        n_channels = trnX.shape[-1]
        n_features = trnX.shape[1:]
        n_classes = len(np.unique(trny))
        hyper = hyper[1:] if hyper is not None else None

        params = {}
        params['n_features'] = n_features
        params['n_classes'] = n_classes
        params['n_channels'] = n_channels

        params['loss_name'] = loss
        params['architecture'] = arch
        params['multigpu'] = multigpu
        params['dataaug'] = dataaug

        params['learning_rate'] = auto_var.get_var("learning_rate")
        params['epochs'] = auto_var.get_var("epochs")
        params['momentum'] = auto_var.get_var("momentum")
        params['optimizer'] = auto_var.get_var("optimizer")
        params['batch_size'] = auto_var.get_var("batch_size")
        params['weight_decay'] = auto_var.get_var("weight_decay")
        params['grad_clip'] = auto_var.get_var("grad_clip")

        model = TorchModel(
            **params,
        )
        return model

