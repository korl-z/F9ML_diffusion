import logging
import torch
from typing import Optional, Dict, Any

# common imports
from ml.common.nn.modules import Module

# custom imports
from ml.diffusion.EDM.model import EDMPrecond, EDMPrecond2
from ml.diffusion.EDM.losses import EDMLoss, EDM2Loss


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
        # x0 = x0.to(self.device) #not needed?
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
        # x0 = x0.to(self.device) #not needed?
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


class LightningVP(Module):
    def __init__(
        self,
        model_conf: Dict[str, Any],
        training_conf: Dict[str, Any],
        model: Optional[torch.nn.Module] = None,
        loss_func: Optional[object] = None,
        tracker: Optional[object] = None,
        split_idx_dct: Optional[Dict[str, Any]] = None,
        scalers: Optional[object] = None,
        selection: Optional[object] = None,
    ):
        super().__init__(
            model_conf=model_conf,
            training_conf=training_conf,
            model=model,
            loss_func=None,
            tracker=tracker,
            split_idx_dct=split_idx_dct,
            scalers=scalers,
            selection=selection,
        )
        if model is None:
            raise RuntimeError("model required")
        if loss_func is None:
            raise RuntimeError("loss_func required")
        self.model = model
        self.vp_loss = loss_func

    def forward(self, batch):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        return self.model(x)

    def _get_loss(self, batch):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
            labels = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            labels = None
        loss = self.vp_loss(self.model, images, labels)
        return loss

    def training_step(self, batch, *args):
        loss = self._get_loss(batch)
        self.log(
            "train_loss",
            loss,
            batch_size=(
                batch[0].size(0) if isinstance(batch, (list, tuple)) else batch.size(0)
            ),
        )
        return loss

    def validation_step(self, batch, *args):
        loss = self._get_loss(batch)
        self.log(
            "val_loss",
            loss,
            batch_size=(
                batch[0].size(0) if isinstance(batch, (list, tuple)) else batch.size(0)
            ),
        )
        return loss

    def test_step(self, batch, *args):
        loss = self._get_loss(batch)
        self.log(
            "test_loss",
            loss,
            batch_size=(
                batch[0].size(0) if isinstance(batch, (list, tuple)) else batch.size(0)
            ),
        )
        return loss
