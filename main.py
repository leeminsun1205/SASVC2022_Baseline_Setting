import argparse
import json
import os
import pickle as pk
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import các thành phần cần thiết
from ECAPATDNN.model import ECAPA_TDNN
from aasist.models.AASIST import Model as AASISTModel
# Sửa lại để import được các hàm từ utils.py
from utils import generate_spk_meta, get_unique_files_from_trial, load_parameters
from vlsp_dataset import VLSPDataset

warnings.filterwarnings("ignore", category=FutureWarning)


def embedding_pipeline(config: DictConfig):
    """
    Hàm chính để kiểm tra và trích xuất embedding nếu cần thiết.
    Nó sẽ kiểm tra các file trong đường dẫn được chỉ định bởi sasv_eval_trial.
    """
    print("--- 🕵️ Bắt đầu kiểm tra và cập nhật embedding ---")
    eval_trial_path = config.dirs.sasv_eval_trial
    # Thư mục chứa file public test
    public_test_base_dir = "/kaggle/input/vlsp-vsasv-public-test/vlsp2025/vlsp2025/"

    required_files = get_unique_files_from_trial(eval_trial_path)
    if not required_files:
        print(f"-> Không tìm thấy file trial tại '{eval_trial_path}' hoặc file rỗng. Bỏ qua.")
        return

    # Tải các embedding hiện có của tập eval
    asv_embd_path = Path(config.dirs.embedding) / "asv_embd_eval.pk"
    cm_embd_path = Path(config.dirs.embedding) / "cm_embd_eval.pk"
    
    asv_embs = {}
    cm_embs = {}
    if os.path.exists(asv_embd_path):
        with open(asv_embd_path, 'rb') as f: asv_embs = pk.load(f)
    if os.path.exists(cm_embd_path):
        with open(cm_embd_path, 'rb') as f: cm_embs = pk.load(f)

    # Tìm những file bị thiếu
    missing_files = [f for f in required_files if f not in asv_embs]
    
    if not missing_files:
        print("-> 🎉 Tất cả các file cần thiết đã có embedding. Sẵn sàng để chạy!")
        return
        
    print(f"-> ❗️ Phát hiện {len(missing_files)} file chưa có embedding. Bắt đầu trích xuất...")

    # --- Khởi tạo mô hình và chạy embedding cho các file bị thiếu ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open("./aasist/config/AASIST.conf", "r") as f_json:
        aasist_cfg = json.loads(f_json.read())
    cm_embd_ext = AASISTModel(aasist_cfg["model_config"]).to(device)
    load_parameters(cm_embd_ext.state_dict(), "./aasist/models/weights/AASIST.pth")
    cm_embd_ext.eval()

    asv_embd_ext = ECAPA_TDNN(C=1024).to(device)
    load_parameters(asv_embd_ext.state_dict(), "./ECAPATDNN/exps/pretrain.model")
    asv_embd_ext.eval()

    dataset = VLSPDataset(file_paths=missing_files, base_dir=public_test_base_dir)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.loader.n_workers)

    new_asv_embs, new_cm_embs = {}, {}
    for batch_x, key in tqdm(loader, desc="Embedding file mới"):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_cm, _ = cm_embd_ext(batch_x)
            batch_asv = asv_embd_ext(batch_x, aug=False)
            for k, cm, asv in zip(key, batch_cm.cpu().numpy(), batch_asv.cpu().numpy()):
                new_asv_embs[k] = asv
                new_cm_embs[k] = cm

    # Cập nhật và lưu lại file .pk
    asv_embs.update(new_asv_embs)
    cm_embs.update(new_cm_embs)
    with open(asv_embd_path, 'wb') as f: pk.dump(asv_embs, f)
    with open(cm_embd_path, 'wb') as f: pk.dump(cm_embs, f)
    
    print(f"--- ✅ Đã cập nhật xong file embedding cho tập eval. ---")


def main(args):
    config = OmegaConf.load(args.config)
    output_dir = Path(args.output_dir)
    pl.seed_everything(config.seed, workers=True)

    # Nếu đang ở chế độ test, hãy kiểm tra và embedding nếu cần
    if args.action == "test":
        embedding_pipeline(config)

    # configure paths
    model_tag = os.path.splitext(os.path.basename(args.config))[0]
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    model_save_path.mkdir(parents=True, exist_ok=True)
    if args.action == "train":
        copy(args.config, model_tag / "config.conf")

    _system = import_module("systems.{}".format(config.pl_system))
    _system = getattr(_system, "System")
    system = _system(config)

    # Configure logging and callbacks
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=model_tag, version=1, name="tsbd_logs"),
        pl.loggers.csv_logs.CSVLogger(save_dir=model_tag, version=1, name="csv_logs"),
    ]
    callbacks = [
        pl.callbacks.ModelSummary(max_depth=3),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=config.progbar_refresh),
    ]
    if args.action == "train":
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=model_save_path,
                filename="{epoch}-{sasv_eer_dev:.5f}",
                monitor="sasv_eer_dev",
                mode="min",
                every_n_epochs=config.val_interval_epoch,
                save_top_k=config.save_top_k,
            )
        )

    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        # PyTorch Lightning sẽ tự động sử dụng tất cả các GPU có sẵn
        # khi devices = -1 hoặc devices = số GPU bạn muốn
        devices=config.ngpus, 
        fast_dev_run=config.fast_dev_run,
        gradient_clip_val=config.gradient_clip
        if config.gradient_clip is not None
        else 0,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        logger=logger,
        max_epochs=config.epoch,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=config.loader.reload_every_n_epoch
        if config.loader.reload_every_n_epoch is not None
        else config.epoch,
        strategy="ddp",
        sync_batchnorm=True,
        val_check_interval=1.0,
    )

    if args.action == "train":
        trainer.fit(system)
        trainer.test(system, ckpt_path="best")
    elif args.action == "test":
        if args.checkpoint_path is None:
            raise ValueError("Vui lòng cung cấp đường dẫn đến checkpoint với cờ --checkpoint_path khi chạy test.")
        trainer.test(system, ckpt_path=args.checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSASV 2025 Baseline framework.")
    parser.add_argument("--config", type=str, required=True, help="configuration file")
    parser.add_argument("--action", type=str, choices=["train", "test"], required=True, help="Hành động cần thực hiện: 'train' hoặc 'test'")
    parser.add_argument("--output_dir", type=str, default="./exp_result", help="Thư mục lưu kết quả")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Đường dẫn đến model checkpoint để chạy test")
    
    main(parser.parse_args())
