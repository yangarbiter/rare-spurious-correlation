import logging
from functools import partial
from typing import Optional

import numpy as np
from autovar import AutoVar
from autovar.base import RegisteringChoiceType, register_var, VariableClass
from autovar.hooks import check_result_file_exist, save_result_to_file
from autovar.hooks import create_placeholder_file, remove_placeholder_if_error
from autovar.hooks import default_get_file_name as get_file_name

from .datasets import DatasetVarClass
from .models import ModelVarClass

auto_var = AutoVar(
    logging_level=logging.INFO,
    before_experiment_hooks=[
        partial(check_result_file_exist, get_name_fn=get_file_name),
        partial(create_placeholder_file, get_name_fn=get_file_name),
    ],
    after_experiment_hooks=[
         partial(save_result_to_file, get_name_fn=get_file_name),
         partial(remove_placeholder_if_error, get_name_fn=get_file_name),
    ],
    settings={
        'file_format': 'pickle',
        'server_url': '',
        'result_file_dir': './results/'
    }
)

auto_var.add_variable_class(DatasetVarClass())
auto_var.add_variable_class(ModelVarClass())
auto_var.add_variable('random_seed', int)
auto_var.add_variable('optimizer', str)
auto_var.add_variable('learning_rate', float, default=1e-2)
auto_var.add_variable('batch_size', int, default=64)
auto_var.add_variable('momentum', float, default=0.9)
auto_var.add_variable('epochs', int, default=2)
auto_var.add_variable('weight_decay', float, default=0.)
auto_var.add_variable('grad_clip', float, default=0.)
auto_var.add_variable('model_path', str, default="")
auto_var.add_variable('gi_depth', int, default=987654321)
auto_var.add_variable('gi_scale', float, default=float("-inf"))

#from autovar.base import RegisteringChoiceType, VariableClass, register_var
#class ExampleVarClass(VariableClass, metaclass=RegisteringChoiceType):
#    """Example Variable Class"""
#    var_name = "example"
#
#    @register_var()
#    @staticmethod
#    def exp(auto_var):
#        pass

#auto_var.add_variable_class(ExampleVarClass())
