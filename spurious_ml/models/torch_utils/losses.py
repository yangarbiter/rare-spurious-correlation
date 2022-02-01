import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_outputs_loss(model, optimizer, base_loss_fn, x, y, **kwargs):
    #loss_name = kwargs['loss_name']
    #batch_size = x.shape[0]

    outputs = model(x)
    loss = base_loss_fn(outputs, y)

    return outputs, loss
