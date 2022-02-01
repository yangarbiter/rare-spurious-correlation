import gc
import os

import torch
from tqdm import tqdm
from torchvision.datasets import VisionDataset
import numpy as np

from .torch_utils.losses import get_outputs_loss
from .torch_utils import data_augs, CustomTensorDataset, get_scheduler, get_loss
from .torch_base import TorchModelBase

DEBUG = int(os.getenv("DEBUG", 0))

class TorchModel(TorchModelBase):
    def __init__(self, *args, **kwargs):
        super(TorchModel, self).__init__(*args, **kwargs)

    def _get_dataset(self, X, y=None, training=True, sample_weights=None,
                     is_img_data=True):
        X = self._preprocess_x(X, is_img_data=is_img_data)
        if sample_weights is None:
            sample_weights = np.ones(len(X))

        if self.dataaug is None:
            transform = None
        else:
            if training == False:
                transform = getattr(data_augs, self.dataaug)()[1]
            else:
                transform = getattr(data_augs, self.dataaug)()[0]

        if training == False:
            if y is None:
                return CustomTensorDataset((torch.from_numpy(X).float(), ), transform=transform)
            else:
                return CustomTensorDataset((torch.from_numpy(X).float(), torch.from_numpy(y).long()), transform=transform)

        dataset = CustomTensorDataset((
             torch.from_numpy(X).float(),
             torch.from_numpy(y).long(),
             torch.from_numpy(sample_weights).float()), transform=transform)
        return dataset

    def fit(self, X, y, unX=None, sample_weights=None, idx_cache_filename=None,
            cache_filename=None, is_img_data=True, with_scheduler=True, verbose=None):
        """
        X, y: nparray
        """
        X = np.asarray(X, dtype=np.float32)
        if unX is None:
            if self.train_type is None:
                pass
            else:
                raise ValueError(f"Unsupported train type: {self.train_type}.")
        else:
            raise NotImplementedError()
            #if self.train_type is not None:
            #    raise ValueError(f"Unlabeled data for training type {self.train_type} not supported.")
            #unlabeled_ds = self._get_dataset(unX, sample_weights=sample_weights, training=True,
            #                                 is_img_data=is_img_data)
        dataset = self._get_dataset(X, y, sample_weights=sample_weights, training=True,
                                    is_img_data=is_img_data)
        return self.fit_dataset(dataset, verbose=verbose,
                is_img_data=is_img_data, with_scheduler=with_scheduler)

    def fit_dataset(self, dataset, unlabeled_ds=None, verbose=None,
            is_img_data=True, with_scheduler=True):
        if verbose is None:
            verbose = 0 if not DEBUG else 1
        log_interval = 1

        history = []
        base_loss_fn = get_loss(self.loss_name, reduction="none")
        scheduler = None
        if with_scheduler:
            scheduler = get_scheduler(self.optimizer, n_epochs=self.epochs, loss_name=self.loss_name)

        train_loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        if unlabeled_ds is not None:
            assert NotImplementedError("not supporting unlabeled data")

        test_loader = None
        if self.tst_ds is not None:
            if isinstance(self.tst_ds, VisionDataset):
                ts_dataset = self.tst_ds
            else:
                tstX, tsty = self.tst_ds
                ts_dataset = self._get_dataset(tstX, tsty, training=False, is_img_data=is_img_data)
            test_loader = torch.utils.data.DataLoader(ts_dataset,
                batch_size=32, shuffle=False, num_workers=self.num_workers)

        for epoch in range(self.start_epoch, self.epochs+1):
            train_loss, train_acc = 0., 0.

            for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
                self.model.train()

                x, y, _ = (d.to(self.device) for d in data)

                self.optimizer.zero_grad()

                #params = {
                #    #'sample_weight': w,
                #    'device': self.device,
                #    'loss_name': self.loss_name,
                #    #'reduction': 'mean',
                #}
                #outputs, loss = get_outputs_loss(
                #    self.model, self.optimizer, base_loss_fn, x, y, **params
                #)
                outputs = self.model(x)
                loss = base_loss_fn(outputs, y)

                loss = loss.mean()
                loss.backward()
                if self.grad_clip != 0.:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                if (epoch - 1) % log_interval == 0:
                    train_loss += loss.item() * len(x)
                    train_acc += (outputs.argmax(dim=1)==y).sum().float().item()

                    self.model.eval()
                    if self.eval_callbacks is not None:
                        for cb_fn in self.eval_callbacks:
                            cb_fn(self.model, train_loader, self.device)

            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            if scheduler:
                scheduler.step()
            self.start_epoch = epoch

            if (epoch - 1) % log_interval == 0:
                print(f"current LR: {current_lr}")
                self.model.eval()
                history.append({
                    'epoch': epoch,
                    'lr': current_lr,
                    'trn_loss': train_loss / len(train_loader.dataset),
                    'trn_acc': train_acc / len(train_loader.dataset),
                })
                print('epoch: {}/{}, train loss: {:.3f}, train acc: {:.3f}'.format(
                    epoch, self.epochs, history[-1]['trn_loss'], history[-1]['trn_acc']))

                if self.tst_ds is not None:
                    tst_loss, tst_acc = self._calc_eval(test_loader, base_loss_fn)
                    history[-1]['tst_loss'] = tst_loss
                    history[-1]['tst_acc'] = tst_acc
                    print('             test loss: {:.3f}, test acc: {:.3f}'.format(
                          history[-1]['tst_loss'], history[-1]['tst_acc']))

        if test_loader is not None:
            del test_loader
        del train_loader
        gc.collect()

        return history
