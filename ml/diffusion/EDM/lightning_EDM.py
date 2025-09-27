import logging
import torch
from typing import Optional, Dict, Any

# common imports
from ml.common.nn.modules import Module

# custom imports
from ml.diffusion.EDM.model import EDMPrecond, EDMPrecond2, VPPrecond, VEPrecond
from ml.diffusion.EDM.losses import EDMLoss, EDM2Loss, VPLoss, VELoss


class EDMModule(Module):
    """
    Lightning Module for training an EDM model (Karras et al., 2022).
    """

    def __init__(
        self,
        datamodule: Any,
        model_conf: Dict[str, Any],
        training_conf: Dict[str, Any],
        data_conf: Optional[Dict[str, Any]] = None,
        model: Optional[torch.nn.Module] = None,
        tracker=None,
    ):
        super().__init__(model_conf, training_conf, model, loss_func=None, tracker=None)
        self.save_hyperparameters(ignore=["model", "tracker"])

        self.datamodule = datamodule
        self.model_conf = model_conf

        # estimate data_sigma from train dataset
        self.datamodule.setup(stage="fit")
        train_data = self.datamodule.train.X.numpy()
        self.sigma_data = train_data.std()
        logging.info(f"[EDMModule] sigma_data from training set: {self.sigma_data:.4f}")

        loss_cfg = self.model_conf["loss_fn"]
        self.loss_fn = EDMLoss(
            P_mean=loss_cfg["P_mean"],
            P_std=loss_cfg["P_std"],
            sigma_data=self.sigma_data,
        )

        self.IMG_SHAPE = tuple(self.hparams.model_conf["img_shape"])

        # wrapper model, always the same
        self.model = EDMPrecond(
            model, sigma_data=self.sigma_data, img_shape=self.IMG_SHAPE
        )

        self.model.sampler_cfg = self.hparams.model_conf["sampler"]

        self._train_losses = []
        self._val_losses = []

    def training_step(self, batch, batch_idx):
        x0, _ = batch
        x0 = x0.to(self.device)
        reshaped_x0 = x0.view(-1, *self.IMG_SHAPE)
        loss_tensor = self.loss_fn(self.model, reshaped_x0)
        loss = loss_tensor.mean()

        self._train_losses.append(loss.detach().cpu().item())
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x0.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x0, _ = batch
        x0 = x0.to(self.device)
        reshaped_x0 = x0.view(-1, *self.IMG_SHAPE)

        loss_tensor = self.loss_fn(self.model, reshaped_x0)
        loss = loss_tensor.mean()

        self._val_losses.append(loss.detach().cpu().item())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x0.size(0),
        )
        return None


class EDM2Module(Module):
    """
    Lightning Module for training an EDM2 model (Karras et al., 2024).
    """

    def __init__(
        self,
        datamodule: Any,
        model_conf: Dict[str, Any],
        training_conf: Dict[str, Any],
        data_conf: Optional[Dict[str, Any]] = None,
        model: Optional[torch.nn.Module] = None,
        tracker=None,
    ):
        super().__init__(model_conf, training_conf, model, tracker)
        self.save_hyperparameters(ignore=["model", "tracker"])

        self.datamodule = datamodule
        self.model_conf = model_conf

        # estimate data_sigma from train dataset
        self.datamodule.setup(stage="fit")
        train_data = self.datamodule.train.X.numpy()
        self.sigma_data = train_data.std()
        logging.info(f"[EDMModule] sigma_data from training set: {self.sigma_data:.4f}")

        loss_cfg = self.model_conf["loss_fn"]
        self.loss_fn = EDM2Loss(
            P_mean=loss_cfg["P_mean"],
            P_std=loss_cfg["P_std"],
            sigma_data=self.sigma_data,
        )

        self.IMG_SHAPE = tuple(self.hparams.model_conf["img_shape"])

        # wrapper model, always the same
        self.model = EDMPrecond2(
            model, sigma_data=self.sigma_data, img_shape=self.IMG_SHAPE
        )

        self.model.sampler_cfg = self.hparams.model_conf["sampler"]

        self._train_losses = []
        self._val_losses = []

    def training_step(self, batch, batch_idx):
        x0, _ = batch
        reshaped_x0 = x0.view(-1, *self.IMG_SHAPE)
        loss_tensor = self.loss_fn(self.model, reshaped_x0)
        loss = loss_tensor.mean()

        self._train_losses.append(loss.detach().cpu().item())
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x0.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x0, _ = batch
        reshaped_x0 = x0.view(-1, *self.IMG_SHAPE)

        loss_tensor = self.loss_fn(self.model, reshaped_x0)
        loss = loss_tensor.mean()

        self._val_losses.append(loss.detach().cpu().item())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x0.size(0),
        )
        return None



class VPModule(Module):
    """
    Lightning Module for training VP model (Karras et al., 2022).
    """
    def __init__(
        self,
        datamodule: Any,
        model_conf: Dict[str, Any],
        training_conf: Dict[str, Any],
        data_conf: Optional[Dict[str, Any]] = None,
        model: Optional[torch.nn.Module] = None,
        tracker=None,
    ):
        super().__init__(model_conf, training_conf, model, tracker)
        self.save_hyperparameters(ignore=["model", "tracker"])

        self.datamodule = datamodule
        self.model_conf = model_conf

        loss_cfg = self.model_conf["loss_fn"]
        self.loss_fn = VPLoss(
            beta_d=loss_cfg["beta_d"],
            beta_min=loss_cfg["beta_min"],
            epsilon_t=loss_cfg["epsilon_t"],
        )

        self.IMG_SHAPE = tuple(self.hparams.model_conf["img_shape"])

        # wrapper model, always the same
        self.model = VPPrecond(
            model, img_shape=self.IMG_SHAPE, M=self.model_conf["M"]
        )

        self.model.sampler_cfg = self.hparams.model_conf["sampler"]

        self._train_losses = []
        self._val_losses = []

    def training_step(self, batch, batch_idx):
        x0, _ = batch
        reshaped_x0 = x0.view(-1, *self.IMG_SHAPE)
        loss_tensor = self.loss_fn(self.model, reshaped_x0)
        loss = loss_tensor.mean()

        self._train_losses.append(loss.detach().cpu().item())
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x0.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x0, _ = batch
        reshaped_x0 = x0.view(-1, *self.IMG_SHAPE)

        loss_tensor = self.loss_fn(self.model, reshaped_x0)
        loss = loss_tensor.mean()

        self._val_losses.append(loss.detach().cpu().item())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x0.size(0),
        )
        return None
    

#VE formulation
class VEModule(Module):
    """
    Lightning Module for training VP model (Karras et al., 2022).
    """
    def __init__(
        self,
        datamodule: Any,
        model_conf: Dict[str, Any],
        training_conf: Dict[str, Any],
        data_conf: Optional[Dict[str, Any]] = None,
        model: Optional[torch.nn.Module] = None,
        tracker=None,
    ):
        super().__init__(model_conf, training_conf, model, tracker)
        self.save_hyperparameters(ignore=["model", "tracker"])

        self.datamodule = datamodule
        self.model_conf = model_conf

        loss_cfg = self.model_conf["loss_fn"]
        self.loss_fn = VELoss(
            sigma_max=loss_cfg["sigma_max"],
            sigma_min=loss_cfg["sigma_min"],
        )

        self.IMG_SHAPE = tuple(self.hparams.model_conf["img_shape"])

        # wrapper model, always the same
        self.model = VEPrecond(
            model, img_shape=self.IMG_SHAPE
        )

        self.model.sampler_cfg = self.hparams.model_conf["sampler"]

        self._train_losses = []
        self._val_losses = []

    def training_step(self, batch, batch_idx):
        x0, _ = batch
        reshaped_x0 = x0.view(-1, *self.IMG_SHAPE)
        loss_tensor = self.loss_fn(self.model, reshaped_x0)
        loss = loss_tensor.mean()

        self._train_losses.append(loss.detach().cpu().item())
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x0.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x0, _ = batch
        reshaped_x0 = x0.view(-1, *self.IMG_SHAPE)

        loss_tensor = self.loss_fn(self.model, reshaped_x0)
        loss = loss_tensor.mean()

        self._val_losses.append(loss.detach().cpu().item())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x0.size(0),
        )
        return None

