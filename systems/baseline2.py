import math
import os
import pickle as pk
from importlib import import_module
from typing import Any

from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
import schedulers as lr_schedulers
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloaders.backend_fusion import collate_fn

from metrics import get_all_EERs
from utils import keras_decay


class System(pl.LightningModule):
    
    # ========== 1. Khởi tạo & cấu hình ==========
    
    def __init__( self, config: DictConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        _model = import_module("models.{}".format(config.model_arch))
        _model = getattr(_model, "Model")
        self.model = _model(config.model_config)
        self.configure_loss()
        self.save_hyperparameters()

        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out
        
    def setup(self, stage=None):
        self.load_meta_information()
        self.load_embeddings()

        if stage in ("fit", "validate") or stage is None:
            module = import_module("dataloaders." + self.config.dataloader)
            self.ds_func_trn = getattr(module, "get_trnset")
            self.ds_func_dev = getattr(module, "get_dev_evalset")
        elif stage == "test":
            module = import_module("dataloaders." + self.config.dataloader)
            self.ds_func_eval = getattr(module, "get_dev_evalset")
        else:
            raise NotImplementedError(".....")
        
    def configure_optimizers(self):
        if self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.wd,
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=self.config.optim.lr,
                momentum=self.config.optim.momentum,
                weight_decay=self.config.optim.wd,
            )
        else:
            raise NotImplementedError("....")

        if self.config.optim.scheduler.lower() == "sgdr_cos_anl":
            assert (
                self.config.optim.n_epoch_per_cycle is not None
                and self.config.optim.min_lr is not None
                and self.config.optim.warmup_steps is not None
                and self.config.optim.lr_mult_after_cycle is not None
            )
            lr_scheduler = lr_schedulers.CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=len(self.train_dataloader())
                // self.config.ngpus
                * self.config.optim.n_epoch_per_cycle,
                cycle_mult=1.0,
                max_lr=self.config.optim.lr,
                min_lr=self.config.optim.min_lr,
                warmup_steps=self.config.optim.warmup_steps,
                gamma=self.config.optim.lr_mult_after_cycle,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        elif self.config.optim.scheduler.lower() == "reduce_on_plateau":
            assert (
                self.config.optim.lr is not None
                and self.config.optim.min_lr is not None
                and self.config.optim.factor is not None
                and self.config.optim.patience is not None
            )
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.config.optim.factor,
                patience=self.config.optim.patience,
                min_lr=self.config.optim.min_lr,
                # verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                    "monitor": "sasv_eer_dev",
                },
            }
        elif self.config.optim.scheduler.lower() == "keras":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: keras_decay(step)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "strict": True,
                },
            }

        else:
            raise NotImplementedError(".....")
    
    def configure_loss(self):
        if self.config.loss.lower() == "bce":
            self.loss = F.binary_cross_entropy_with_logits
        if self.config.loss.lower() == "cce":
            self.loss = torch.nn.CrossEntropyLoss(
                weight=torch.FloatTensor(self.config.loss_weight)
            )
        else:
            raise NotImplementedError("!")
        
    def load_meta_information(self):
        with open(self.config.dirs.spk_meta + "spk_meta_trn.pk", "rb") as f:
            self.spk_meta_trn = pk.load(f)
        with open(self.config.dirs.spk_meta + "spk_meta_dev.pk", "rb") as f:
            self.spk_meta_dev = pk.load(f)
        with open(self.config.dirs.spk_meta + "spk_meta_eval.pk", "rb") as f:
            self.spk_meta_eval = pk.load(f)
            
    def load_embeddings(self):
        # load saved countermeasures(CM) related preparations
        with open(self.config.dirs.embedding + "cm_embd_trn.pk", "rb") as f:
            self.cm_embd_trn = pk.load(f)
        with open(self.config.dirs.embedding + "cm_embd_dev.pk", "rb") as f:
            self.cm_embd_dev = pk.load(f)
        with open(self.config.dirs.embedding + "cm_embd_eval.pk", "rb") as f:
            self.cm_embd_eval = pk.load(f)
        with open(self.config.dirs.embedding + "cm_embd_public_test.pk", "rb") as f:
            self.cm_embd_public_test = pk.load(f)

        # load saved automatic speaker verification(ASV) related preparations
        with open(self.config.dirs.embedding + "asv_embd_trn.pk", "rb") as f:
            self.asv_embd_trn = pk.load(f)
        with open(self.config.dirs.embedding + "asv_embd_dev.pk", "rb") as f:
            self.asv_embd_dev = pk.load(f)
        with open(self.config.dirs.embedding + "asv_embd_eval.pk", "rb") as f:
            self.asv_embd_eval = pk.load(f)
        with open(self.config.dirs.embedding + "asv_embd_public_test.pk", "rb") as f:
            self.asv_embd_public_test = pk.load(f)
    
    # ========== 2. Dataloader ==========
    
    def train_dataloader(self):
        self.train_ds = self.ds_func_trn(self.cm_embd_trn, self.asv_embd_trn, self.spk_meta_trn)
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.loader.n_workers,
        )

    def val_dataloader(self):
        with open(self.config.dirs.sasv_dev_trial, "r") as f:
            sasv_dev_trial = f.readlines()
        
        self.dev_ds = self.ds_func_dev(
            sasv_dev_trial, self.cm_embd_dev, self.asv_embd_dev
        )
        return DataLoader(
            self.dev_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.loader.n_workers,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        with open(self.config.dirs.sasv_eval_trial, "r") as f:
            sasv_eval_trial = f.readlines()

        self.eval_ds = self.ds_func_eval(
            sasv_eval_trial, self.cm_embd_eval, self.asv_embd_eval
        )
        return DataLoader(
            self.eval_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.loader.n_workers,
            collate_fn=collate_fn
        )

    # ========== 3. Training ==========
    
    def training_step(self, batch, batch_idx):
        if batch[0] is None:
            return None
            
        embd_asv_enrol, embd_asv_test, embd_cm_test, label = batch
        pred = self.model(embd_asv_enrol, embd_asv_test, embd_cm_test)
        loss = self.loss(pred, label)
        self.log(
            "trn_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss
    
    # ========== 4. Validation ==========
    
    def validation_step(self, batch, batch_idx):
        if batch[0] is None:
            return None
            
        embd_asv_enrol, embd_asv_test, embd_cm_test, key = batch
        pred = self.model(embd_asv_enrol, embd_asv_test, embd_cm_test)
        pred = torch.softmax(pred, dim=-1)
        output = {"pred": pred, "key": key}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            print("Cảnh báo: Không có output nào trong vòng validation để xử lý.")
            return

        log_dict = {}
        preds, keys = [], []
        for output in self.validation_step_outputs:
            preds.append(output["pred"])
            keys.extend(list(output["key"]))

        preds = torch.cat(preds, dim=0)[:, 1].detach().cpu().numpy()
        sasv_eer, sv_eer, spf_eer = get_all_EERs(preds=preds, keys=keys)

        log_dict["sasv_eer_dev"] = sasv_eer
        log_dict["sv_eer_dev"] = sv_eer
        log_dict["spf_eer_dev"] = spf_eer

        self.log_dict(log_dict)
        self.validation_step_outputs.clear()

    # ========== 5. Testing ==========
    
    def test_step(self, batch, batch_idx):
        if batch[0] is None:
            return None
            
        # Tái sử dụng logic của validation_step
        res_dict = self.validation_step(batch, batch_idx)
        if res_dict:
            # Ghi đè list lưu trữ để không bị lẫn lộn
            self.validation_step_outputs.pop() # Xóa output vừa được thêm bởi validation_step
            self.test_step_outputs.append(res_dict)
        return res_dict

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            print("Cảnh báo: Không có output nào trong vòng test để xử lý.")
            return

        log_dict = {}
        preds, keys = [], []
        for output in self.test_step_outputs:
            preds.append(output["pred"])
            keys.extend(list(output["key"]))

        preds = torch.cat(preds, dim=0)[:, 1].detach().cpu().numpy()
        sasv_eer, sv_eer, spf_eer = get_all_EERs(preds=preds, keys=keys)

        log_dict["sasv_eer_eval"] = sasv_eer
        log_dict["sv_eer_eval"] = sv_eer
        log_dict["spf_eer_eval"] = spf_eer

        self.log_dict(log_dict)
        self.test_step_outputs.clear()

