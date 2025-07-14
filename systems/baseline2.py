import math
import os
import pickle as pk
from importlib import import_module
from typing import Any

import omegaconf
import pytorch_lightning as pl
import schedulers as lr_schedulers
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Dòng import bị thiếu
from dataloaders.backend_fusion import collate_fn

from metrics import get_all_EERs
from utils import keras_decay


class System(pl.LightningModule):
    def __init__(
        self, config: omegaconf.dictconfig.DictConfig, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        _model = import_module("models.{}".format(config.model_arch))
        _model = getattr(_model, "Model")
        self.model = _model(config.model_config)
        self.configure_loss()
        self.save_hyperparameters()

        # Khởi tạo thuộc tính để lưu output
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        # Kiểm tra xem batch có hợp lệ không (do collate_fn có thể trả về None)
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

    def test_step(self, batch, batch_idx):
        if batch[0] is None: return None
        
        embd_asv_enrol, embd_asv_test, embd_cm_test, key = batch
        pred = self.model(embd_asv_enrol, embd_asv_test, embd_cm_test)
        score = torch.softmax(pred, dim=-1)[:, 1]

        enroll_paths, test_paths, ans_keys = key
        
        output = {
            "score": score, 
            "enroll_path": enroll_paths, 
            "test_path": test_paths, 
            "key": ans_keys
        }
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            print("Cảnh báo: Không có output nào trong vòng test để xử lý.")
            return

        submission_file_path = "submission.txt"
        print(f"\n✍️  Bắt đầu ghi file submission vào: {submission_file_path}")
        
        with open(submission_file_path, "w") as f:
            f.write("enrollment_wav\ttest_wav\tscore\n")
            for output in self.test_step_outputs:
                scores, enrolls, tests = output["score"], output["enroll_path"], output["test_path"]
                for i in range(len(scores)):
                    f.write(f"{enrolls[i]}\t{tests[i]}\t{scores[i].item()}\n")
                    
        print(f"-> ✅ Đã ghi xong file submission!")

        # Phần tính EER chỉ chạy nếu có nhãn hợp lệ
        keys, preds = [], []
        has_valid_labels = any(k != 'public_test' for output in self.test_step_outputs for k in output["key"])

        if has_valid_labels:
            print("\n📊 Phát hiện có nhãn hợp lệ, đang tính toán EER...")
            for output in self.test_step_outputs:
                valid_indices = [i for i, k in enumerate(output["key"]) if k != 'public_test']
                if valid_indices:
                    preds.append(output["score"][valid_indices])
                    keys.extend([output["key"][i] for i in valid_indices])
            
            if preds:
                log_dict = {}
                preds = torch.cat(preds, dim=0).detach().cpu().numpy()
                sasv_eer, sv_eer, spf_eer = get_all_EERs(preds=preds, keys=keys)
                log_dict["sasv_eer_eval"] = sasv_eer
                self.log_dict(log_dict)
                print(f"-> EER trên tập đánh giá có nhãn: {sasv_eer:.4f}")

        self.test_step_outputs.clear()


    def configure_optimizers(self):
        # ... (Phần này bạn đã làm đúng, giữ nguyên)
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
                verbose=True,
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
        
        # SỬA LẠI HOÀN CHỈNH Ở ĐÂY
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

        # SỬA LẠI HOÀN CHỈNH Ở ĐÂY
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

    def configure_loss(self):
        # ... (Phần này bạn đã làm đúng, giữ nguyên)
        if self.config.loss.lower() == "bce":
            self.loss = F.binary_cross_entropy_with_logits
        if self.config.loss.lower() == "cce":
            self.loss = torch.nn.CrossEntropyLoss(
                weight=torch.FloatTensor(self.config.loss_weight)
            )
        else:
            raise NotImplementedError("!")

    def load_meta_information(self):
        # ... (Phần này bạn đã làm đúng, giữ nguyên)
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

        # load saved automatic speaker verification(ASV) related preparations
        with open(self.config.dirs.embedding + "asv_embd_trn.pk", "rb") as f:
            self.asv_embd_trn = pk.load(f)
        with open(self.config.dirs.embedding + "asv_embd_dev.pk", "rb") as f:
            self.asv_embd_dev = pk.load(f)
        with open(self.config.dirs.embedding + "asv_embd_eval.pk", "rb") as f:
            self.asv_embd_eval = pk.load(f)
