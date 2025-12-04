import torch
from typing import Optional, Dict, Any
import logging

#custom imports
from ml.diffusion.ddpm.model import DDPMPrecond, iDDPMPrecond, iDDPM2Precond
from ml.diffusion.ddpm.diffusers import DiffuserDDPMeps, DiffuseriDDPMeps
from ml.diffusion.ddpm.losses import DDPMLoss, HybridLoss, iDDPMloss
from ml.common.nn.modules import Module


class DDPMModule(Module):
    """
    Lightning Module for training a DDPM noise-prediction model.
    Accepts a pre-configured model. Diffuser and loss built in manually.
    """

    def __init__(
        self,
        datamodule: Any,
        model_conf: Dict[str, Any],
        training_conf: Dict[str, Any],
        data_conf: Dict[str, Any],
        model: Optional[torch.nn.Module] = None,
        tracker=None,
    ):
        super().__init__(model_conf, training_conf, model, tracker)
        self.save_hyperparameters(ignore=["model", "diffuser", "tracker"])

        self.model_conf = model_conf

        # instantiate model, diffuser, loss
        self.model = model
        self.diffuser = DiffuserDDPMeps(
            timesteps=self.model_conf["diffuser"]["timesteps"],
            scheduler=self.model_conf["diffuser"]["scheduler"],
            device="cuda",
        ) 
        self.loss_fn = DDPMLoss(diffuser=self.diffuser)

        # wrapper model, always the same
        self.model = DDPMPrecond(
            model,
            data_dim=data_conf["input_dim"],
            diffuser=self.diffuser,
        )

        self.model.set_diffuser(self.diffuser)

    def training_step(self, batch, batch_idx):
        x0, _ = batch
        loss = self.loss_fn(self.model, x0)
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
        loss = self.loss_fn(self.model, x0)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x0.size(0),
        )


class iDDPMModule(Module):
    """
    Lightning Module for training a iDDPM noise-prediction model.
    """

    def __init__(
        self,
        datamodule: Any,
        model_conf: Dict[str, Any],
        training_conf: Dict[str, Any],
        data_conf: Dict[str, Any],
        model: Optional[torch.nn.Module] = None,
        tracker=None,
    ):
        super().__init__(model_conf, training_conf, model, tracker)
        self.save_hyperparameters(ignore=["model", "diffuser", "tracker"])

        self.model = model
        self.diffuser = DiffuseriDDPMeps(
            timesteps=self.model_conf["diffuser"]["timesteps"],
            scheduler=self.model_conf["diffuser"]["scheduler"],
            device="cuda",
        )  # device fixed to gpu for now
        self.loss_fn = HybridLoss(
            diffuser=self.diffuser, vlb_weight=model_conf["vlb_weight"]
        )
        logging.info(f"using vlb_weight={model_conf['vlb_weight']}")

        # wrapper model, always the same
        self.model = iDDPMPrecond(
            model,
            data_dim=data_conf["input_dim"],
            diffuser=self.diffuser,
        )

        self.model.set_diffuser(self.diffuser)

    def training_step(self, batch, batch_idx):
        x0, _ = batch
        loss = self.loss_fn(self.model, x0)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=x0.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x0, _ = batch
        loss = self.loss_fn(self.model, x0)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x0.size(0),
        )


class iDDPM2Module(Module):
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
        super().__init__(model_conf, training_conf, model, tracker)
        self.save_hyperparameters(ignore=["model", "tracker"])

        self.model_conf = model_conf

        self.IMG_SHAPE = tuple(self.hparams.model_conf["img_shape"])

        # wrapper model, always the same
        self.model = iDDPM2Precond(model, img_shape=self.IMG_SHAPE)

        self.loss_fn = iDDPMloss(u_buffer=self.model.u)

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
