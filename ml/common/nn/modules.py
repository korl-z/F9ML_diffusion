
import copy
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import lightning as L
import torch

class Tracker(ABC):
    def __init__(self, experiment_conf, tracker_path):
        self.experiment_conf = experiment_conf
        self.tracker_path = tracker_path

        self.module = None

        self.current_epoch = None  # current epoch number
        self.plotting_dirs = None  # directories for saving plots
        self.stage = None  # set in get_predictions

        self.base_dir = f"{self.tracker_path}/{self.experiment_conf['run_name']}/"
        logging.debug(f"Tracker base directory: {self.base_dir}")

    def __call__(self, module):
        self.module = module
        return self

    def on_first_epoch(self):
        """Create directories if they don't exist yet, should be called after the first epoch in compute method."""
        self.create_dirs()

    def create_dirs(self):
        """Creates the directories where the plots will be saved."""
        self.plotting_dirs = self.make_plotting_dirs()

        # create directories if they don't exist yet
        for d in list(self.plotting_dirs.values()):
            if not os.path.exists(d):
                logging.debug(f"Creating tracker directory after first epoch: {d}")
                os.makedirs(d)

    @abstractmethod
    def make_plotting_dirs(self):
        """Create a dictionary of directories for different plotting graphs."""
        pass

    @abstractmethod
    def get_predictions(self, stage):
        """Needs to be implemented for different tasks. Basically, it is the forward of the model."""
        return None

    @abstractmethod
    def compute(self, stage):
        self.stage, self.current_epoch = stage, self.module.current_epoch

        if self.current_epoch == 0:
            self.on_first_epoch()

        # check if metrics should be calculated this epoch
        if self.current_epoch % self.experiment_conf["check_metrics_n_epoch"] != 0 and self.stage != "test":
            logging.debug(f"Skipping metrics computation for epoch {self.current_epoch}")
            return False

        # get predictions, needs to be implemented
        self.get_predictions(stage)

        return True

    @abstractmethod
    def plot(self):
        """Plot the metrics."""
        self.module.logger.experiment.log_artifact(local_path=self.base_dir, run_id=self.module.logger.run_id)
        return None


class Module(L.LightningModule):
    """
    Generic base Lightning module (thin wrapper).
    Accepts `model_conf` and `training_conf` dicts and a torch `model`.
    If `loss_func` is provided, `_get_loss` will call it with (targets, preds).
    """

    def __init__(
        self,
        model_conf: Dict[str, Any],
        training_conf: Dict[str, Any],
        model: Optional[torch.nn.Module] = None,
        loss_func=None,
        tracker=None,
        split_idx_dct=None,
        scalers=None,
        selection=None,
    ):
        super().__init__()
        self.model_conf = model_conf
        self.training_conf = training_conf
        self.loss_func = loss_func
        self.tracker = tracker
        
        self.split_idx_dct = split_idx_dct
        self.scalers = scalers
        self.selection = selection

        self.model = model
        self.uncompiled_model = None

        if self.training_conf.get("compile", False) and (self.model is not None):
            logging.info("[b][red]Torch compile is ON! Model will be compiled in default mode.[/red][/b]")
            self.uncompiled_model = self.model
            self.model = torch.compile(self.model, mode="default")

    def configure_optimizers(self):
        """
        training_conf expected structure:
          training_conf['optimizer'] = {'name': 'Adam', 'lr': 1e-3, 'weight_decay': 1e-5}
          training_conf['scheduler'] = {'use': False}
        """
        opt_conf = self.training_conf["optimizer"]
        name = opt_conf["name"]
        lr = float(opt_conf["lr"])
        wd = float(opt_conf["weight_decay"])
        optimizer_cls = getattr(torch.optim, name)
        optimizer = optimizer_cls(self.model.parameters(), lr=lr, weight_decay=wd)

        sched_conf = self.training_conf["scheduler"]
        if sched_conf.get("use", False):
            sched_name = sched_conf["scheduler_name"]
            sched_params = sched_conf["scheduler_params"]
            sched_cls = getattr(torch.optim.lr_scheduler, sched_name)
            scheduler = sched_cls(optimizer, **sched_params)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": sched_conf.get("monitor", "val_loss"),
                    "interval": sched_conf.get("interval", "epoch"),
                },
            }
        return optimizer

    def forward(self, batch):
        """
        Default forward expects input (X, y).
        """
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        return self.model(x)

    def _get_loss(self, batch):
        """
        Default supervised loss path: expects batch=(x,y) and self.loss_func signature (y, y_pred).
        """
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            raise RuntimeError("Expected batch=(x,y)")
        preds = self.forward(batch)
        loss = self.loss_func(y, preds)
        return loss

    def training_step(self, batch, *args):
        loss = self._get_loss(batch)
        self.log("train_loss", loss, batch_size=batch[0].size()[0])
        return loss

    def validation_step(self, batch, *args):
        loss = self._get_loss(batch)
        self.log("val_loss", loss, batch_size=batch[0].size()[0])
        return loss

    def test_step(self, batch, *args):
        loss = self._get_loss(batch)
        self.log("test_loss", loss, batch_size=batch[0].size()[0])
        return loss

    def on_train_start(self):
        try:
            dm = self.trainer.datamodule
            if dm is not None:
                if hasattr(dm, "scalers"):
                    self.scalers = dm.scalers
                if hasattr(dm, "train_idx"):
                    self.split_idx_dct = {
                        "train_idx": getattr(dm, "train_idx", None),
                        "val_idx": getattr(dm, "val_idx", None),
                        "test_idx": getattr(dm, "test_idx", None),
                    }
        except Exception as e:
            logging.debug(
                "on_train_start: could not extract datamodule attributes: %s", e
            )

        try:
            run_id = getattr(self.logger, "run_id", None)
            client = getattr(self.logger, "experiment", None)
            if client is not None and run_id is not None:
                import mlflow

                with mlflow.start_run(run_id=run_id):
                    mlflow.log_text(str(self), artifact_file="model_str.txt")
        except Exception:
            pass
        
    def on_validation_epoch_end(self):
        if self.tracker:
            self.tracker.compute(stage="val")
            self.tracker.plot()

    def on_test_start(self):
        if self.tracker:
            self.tracker.compute(stage="test")
            self.tracker.plot()