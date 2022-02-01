import os

import torch
import torch.nn.functional as F
from torchvision.datasets import VisionDataset

import numpy as np
from sklearn.base import BaseEstimator
from .torch_utils import get_optimizer, CustomTensorDataset, archs, data_augs

DEBUG = int(os.getenv("DEBUG", 0))


class TorchModelBase(BaseEstimator):

    def __init__(self, n_features, n_classes, loss_name='ce',
                n_channels=None, learning_rate=1e-4, momentum=0.0, weight_decay=0.0,
                batch_size=256, epochs=20, optimizer='sgd', architecture='arch_001',
                random_state=None, eval_callbacks=None, train_type=None, grad_clip: float=0.,
                norm=np.inf, multigpu=False, dataaug=None, device=None, num_workers=4):
        print(f'lr: {learning_rate}, opt: {optimizer}, loss: {loss_name}, '
              f'arch: {architecture}, dataaug: {dataaug}, batch_size: {batch_size}, '
              f'momentum: {momentum}, weight_decay: {weight_decay}, grad_clip: {grad_clip}, '
              f'epochs: {epochs}, train_type: {train_type}')
        self.num_workers = num_workers
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.epochs = epochs
        self.loss_name = loss_name
        self.dataaug = dataaug
        self.grad_clip = grad_clip

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        arch_fn = getattr(archs, self.architecture)
        arch_params = dict(n_features=n_features, n_classes=self.n_classes, n_channels=n_channels)
        model = arch_fn(**arch_params).to(self.device)

        self.multigpu = multigpu
        if self.multigpu:
            model = torch.nn.DataParallel(model, device_ids=[0, 1])

        self.optimizer = get_optimizer(model, optimizer, learning_rate, momentum, weight_decay)
        self.model = model

        self.eval_callbacks = eval_callbacks
        self.random_state = random_state
        self.train_type = train_type

        self.tst_ds = None
        self.start_epoch = 1

    def _calc_eval(self, loader, loss_fn):
        cum_loss, cum_acc = 0., 0.
        with torch.no_grad():
            for data in loader:
                tx, ty = data[0], data[1]
                tx, ty = tx.to(self.device), ty.to(self.device)
                outputs = self.model(tx)
                if loss_fn.reduction == 'none':
                    loss = torch.sum(loss_fn(outputs, ty))
                else:
                    loss = loss_fn(outputs, ty)
                cum_loss += loss.item()
                cum_acc += (outputs.argmax(dim=1)==ty).sum().float().item()
        return cum_loss / len(loader.dataset), cum_acc / len(loader.dataset)

    def _preprocess_x(self, X, is_img_data=True):
        if len(X.shape) == 4 and is_img_data == True:
            return X.transpose(0, 3, 1, 2)
        else:
            return X

    #def fit(self, X, y, sample_weights=None, verbose=None, is_img_data=True):
    #    dataset = self._get_dataset(X, y, sample_weights, is_img_data=is_img_data)
    #    return self.fit_dataset(dataset, verbose=verbose)

    def _prep_pred(self, X, is_img_data=True):
        self.model.eval()
        if isinstance(X, VisionDataset):
            dataset = X
        else:
            if self.dataaug is None:
                transform = None
            else:
                transform = getattr(data_augs, self.dataaug)()[1]
            X = self._preprocess_x(X, is_img_data=is_img_data)
            dataset = CustomTensorDataset((torch.from_numpy(X).float(), ), transform=transform)
        loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def get_repr(self, X, is_img_data=True):
        loader = self._prep_pred(X, is_img_data=is_img_data)
        ret = []
        for [x] in loader:
            ret.append(self.model.get_repr(x.to(self.device)).detach().cpu().numpy())
        del loader
        return np.concatenate(ret)

    def predict(self, X, is_img_data=True):
        loader = self._prep_pred(X, is_img_data=is_img_data)
        ret = []
        with torch.no_grad():
            for [x] in loader:
                ret.append(self.model(x.to(self.device)).argmax(1).cpu().numpy())
        del loader
        return np.concatenate(ret)

    def predict_proba(self, X, is_img_data=True):
        loader = self._prep_pred(X, is_img_data=is_img_data)
        ret = []
        with torch.no_grad():
            for [x] in loader:
                output = F.softmax(self.model(x.to(self.device)).detach(), dim=1)
                ret.append(output.cpu().numpy())
        del loader
        return np.concatenate(ret, axis=0)

    def predict_real(self, X, is_img_data=True):
        loader = self._prep_pred(X, is_img_data=is_img_data)
        ret = []
        for [x] in loader:
            x.requires_grad_(False)
            ret.append(self.model(x.to(self.device)).detach().cpu().numpy())
        del loader
        return np.concatenate(ret, axis=0)

    def save(self, path):
        if self.multigpu:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        torch.save({
            'epoch': self.start_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        loaded = torch.load(path)
        self.start_epoch = loaded['epoch']
        self.model.load_state_dict(loaded['model_state_dict'])
        self.optimizer.load_state_dict(loaded['optimizer_state_dict'])
        self.model.eval()

    def predict_ds(self, ds):
        loader = torch.utils.data.DataLoader(ds,
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        ret = []
        for x in loader:
            pred = self.model(x[0].to(self.device)).argmax(1).cpu().numpy()
            ret.append(pred)
        del loader
        return np.concatenate(ret)
