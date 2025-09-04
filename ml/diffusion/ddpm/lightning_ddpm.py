import sys
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import torch
import torch.nn.functional as F
import lightning as L
from typing import Optional, Dict, Any
import mlflow
from sklearn import preprocessing
from lightning.pytorch.callbacks import Callback
import logging

sys.path.insert(1, "C:/Users/Uporabnik/Documents/IJS-F9/korlz")

from ml.diffusion.ddpm.model import NoisePredictorUNet, DDPMPrecond, iDDPMPrecond
from ml.diffusion.ddpm.diffusers import DiffuserDDPMeps
from ml.diffusion.ddpm.losses import DDPMLoss, HybridLoss
from ml.diffusion.ddpm.samplers import SamplerNoise
from ml.common.utils.utils import EMA
from ml.common.nn.modules import Module
from ml.common.data_utils.downloadutils import load_dataset_variables
from ml.common.utils.plot_utils import add_data_mc_ratio


#temporary list for plotting
features_list = [
    "lepton pT",
    "lepton eta",
    "missing energy",
    "jet1 pT",
    "jet1 eta",
    "jet2 pT",
    "jet2 eta",
    "jet3 pT",
    "jet3 eta",
    "jet4 pT",
    "jet4 eta",
    "m jj",
    "m jjj",
    "m lv",
    "m jlv",
    "m bb",
    "m wbb",
    "m wwbb",
]


# class DDPMModule(Module):
#     """
#     Specialized DDPM lightning module (eps-predicting training loop).
#     - model_conf: model params
#     - training_conf: training params
#     - model
#     - diffuser
#     """

#     def __init__(
#         self,
#         datamodule: Any,
#         model_conf: Dict[str, Any],
#         training_conf: Dict[str, Any],
#         data_conf: Optional[Dict[str, Any]] = None,
#         model: Optional[torch.nn.Module] = None,
#         diffuser: Optional[Any] = None,
#         loss_func: Optional[Any] = None,
#         test_sample_count: int = 1024,
#         tracker = None
#     ):
#         super().__init__(model_conf, training_conf, model, loss_func=None, tracker=None)
#         self.save_hyperparameters(ignore=["model", "diffuser", "loss_func", "tracker"])

#         self.diffuser = diffuser
#         self.datamodule = datamodule
#         self.test_sample_count = test_sample_count

#         if loss_func is None:
#             self.loss_fn = LossDDPMNoise(model, diffuser)
#         else:
#             self.loss_fn = loss_func(model, diffuser)

#         self._train_losses = []
#         self._val_losses = []

#         self.train_loss_history = []
#         self.val_loss_history = []

#         self._test_outputs = []

#     def training_step(self, batch, batch_idx):
#         if isinstance(batch, (list, tuple)):
#             x0 = batch[0]
#         else:
#             x0 = batch
#         x0 = x0.to(self.device)

#         loss = self.loss_fn(x0)

#         try:
#             self._train_losses.append(loss.detach().cpu().item())
#         except Exception:
#             pass

#         self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=x0.size(0))

#         return loss

#     def validation_step(self, batch, batch_idx):
#         if isinstance(batch, (list, tuple)):
#             x0 = batch[0]
#         else:
#             x0 = batch
#         x0 = x0.to(self.device)

#         loss = self.loss_fn(x0)

#         try:
#             self._val_losses.append(loss.detach().cpu().item())
#         except Exception:
#             pass

#         self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x0.size(0))

#         return None

#     def test_step(self, batch, batch_idx):
#         if isinstance(batch, (list, tuple)):
#             x0 = batch[0]
#         else:
#             x0 = batch
#         self._test_outputs.append(x0.detach().cpu())
#         return None

#     def on_train_epoch_end(self) -> None:
#         if len(self._train_losses) > 0:
#             epoch_mean = float(np.mean(self._train_losses))
#             self.train_loss_history.append(epoch_mean)
#             self.log("train_epoch_loss", epoch_mean, prog_bar=True)
#         self._train_losses = []

#     def on_validation_epoch_end(self) -> None:
#         if len(self._val_losses) > 0:
#             epoch_mean = float(np.mean(self._val_losses))
#             self.val_loss_history.append(epoch_mean)
#             self.log("val_epoch_loss", epoch_mean, prog_bar=True)
#         self._val_losses = []

#     def on_test_start(self) -> None:
#         self._test_outputs = []

class DDPMModule(Module):
    """
    Lightning Module for training a DDPM noise-prediction model.
    Accepts a pre-configured model and diffuser.
    """
    def __init__(
        self,
        datamodule: Any,
        model_conf: Dict[str, Any],
        training_conf: Dict[str, Any],
        data_conf: Dict[str, Any],
        model: Optional[torch.nn.Module] = None,
        diffuser: Optional[Any] = None, 
        tracker = None
    ):
        super().__init__(model_conf, training_conf, model)
        self.save_hyperparameters(ignore=['model', 'diffuser', 'tracker'])
        
        self.datamodule = datamodule
        self.model_conf = model_conf

        self.model = model
        self.diffuser = diffuser
        self.loss_fn = DDPMLoss(diffuser=self.diffuser)

        # wrapper model, always the same
        self.model = DDPMPrecond(
            model, 
            data_dim=data_conf["input_dim"],
            diffuser = self.diffuser,
        )
        
        self.model.set_diffuser(self.diffuser)


    def training_step(self, batch, batch_idx):
        x0, _ = batch 
        loss = self.loss_fn(self.model, x0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x0.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x0, _ = batch
        loss = self.loss_fn(self.model, x0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x0.size(0))


class iDDPMModule(Module):
    """
    Lightning Module for training a iDDPM noise-prediction model.
    Accepts a pre-configured model and diffuser.
    """
    def __init__(
        self,
        datamodule: Any,
        model_conf: Dict[str, Any],
        training_conf: Dict[str, Any],
        data_conf: Dict[str, Any],
        model: Optional[torch.nn.Module] = None,
        diffuser: Optional[Any] = None, 
        tracker = None
    ):
        super().__init__(model_conf, training_conf, model)
        self.save_hyperparameters(ignore=['model', 'diffuser', 'tracker'])
        
        self.model = model
        self.diffuser = diffuser
        self.loss_fn = HybridLoss(diffuser=self.diffuser, vlb_weight=model_conf["vlb_weight"])
        logging.info(f"using vlb_weight={model_conf['vlb_weight']}")

        # wrapper model, always the same
        self.model = iDDPMPrecond(
            model, 
            data_dim=data_conf["input_dim"],
            diffuser = self.diffuser,
        )
        
        self.model.set_diffuser(self.diffuser)

    def training_step(self, batch, batch_idx):
        x0, _ = batch 
        loss = self.loss_fn(self.model, x0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x0.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x0, _ = batch
        loss = self.loss_fn(self.model, x0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x0.size(0))
