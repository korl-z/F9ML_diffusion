import torch
from typing import Optional, Dict, Any
import logging


from ml.diffusion.ddpm.model import NoisePredictorUNet, DDPMPrecond, iDDPMPrecond, iDDPM2Precond
from ml.diffusion.ddpm.diffusers import DiffuserDDPMeps
from ml.diffusion.ddpm.losses import DDPMLoss, HybridLoss, iDDPMloss
from ml.diffusion.ddpm.samplers import SamplerNoise
from ml.common.nn.modules import Module
from ml.common.nn.unet import MPTinyUNet


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
        tracker = None,
    ):
        super().__init__(model_conf, training_conf, model, loss_func=None, tracker=None)
        self.save_hyperparameters(ignore=["model", "tracker"])

        self.datamodule = datamodule
        self.model_conf = model_conf

        self.IMG_SHAPE = tuple(self.hparams.model_conf["img_shape"])

        # wrapper model, always the same
        self.model = iDDPM2Precond(
            model, 
            img_shape=self.IMG_SHAPE
        )

        self.loss_fn = iDDPMloss(
            u_buffer=self.model.u
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
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x0.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x0, _ = batch
        x0 = x0.to(self.device)
        reshaped_x0 = x0.view(-1, *self.IMG_SHAPE)

        loss_tensor = self.loss_fn(self.model, reshaped_x0)
        loss = loss_tensor.mean()

        self._val_losses.append(loss.detach().cpu().item())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x0.size(0))
        return None