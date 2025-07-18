import argparse
import json
import os
import pickle as pk
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from vlsp_dataset import VLSPDataset
from aasist.models.AASIST import Model as AASISTModel
from ECAPATDNN.model import ECAPA_TDNN
from utils import load_parameters

# Danh sách các phân vùng dữ liệu
SET_PARTITION = ["trn", "dev", "eval"]

# Cập nhật đường dẫn đến các tệp protocol
SET_CM_PROTOCOL = {
    "trn": "protocols/train_split.txt",
    "dev": "protocols/dev_split.txt",
    "eval": "protocols/eval_split.txt",
}

# Thư mục gốc chứa dữ liệu âm thanh
BASE_DIR = "/kaggle/input/vlsp-vsasv-datasets/vlsp2025/"

def save_embeddings(set_name, cm_embd_ext, asv_embd_ext, device):
    """Trích xuất và lưu ASV và CM embeddings."""
    meta_lines = open(SET_CM_PROTOCOL[set_name], "r").readlines()
    utt_list = [line.strip().split(" ")[1] for line in meta_lines]
    dataset = VLSPDataset(file_paths=utt_list, base_dir=BASE_DIR)
    loader = DataLoader(dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True)
    cm_emb_dic, asv_emb_dic = {}, {}

    print(f"Bắt đầu trích xuất embedding cho tập {set_name}...")
    for batch_x, key in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_cm_emb, _ = cm_embd_ext(batch_x)
            batch_cm_emb = batch_cm_emb.detach().cpu().numpy()
            batch_asv_emb = asv_embd_ext(batch_x, aug=False).detach().cpu().numpy()
        for k, cm_emb, asv_emb in zip(key, batch_cm_emb, batch_asv_emb):
            cm_emb_dic[k] = cm_emb
            asv_emb_dic[k] = asv_emb

    os.makedirs("embeddings", exist_ok=True)
    with open(f"embeddings/cm_embd_{set_name}.pk", "wb") as f: pk.dump(cm_emb_dic, f)
    with open(f"embeddings/asv_embd_{set_name}.pk", "wb") as f: pk.dump(asv_emb_dic, f)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-aasist_config", type=str, default="./aasist/config/AASIST.conf")
    parser.add_argument("-aasist_weight", type=str, default="./aasist/models/weights/AASIST.pth")
    parser.add_argument("-ecapa_weight", type=str, default="./ECAPATDNN/exps/pretrain.model")
    return parser.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    with open(args.aasist_config, "r") as f_json: config = json.loads(f_json.read())
    model_config = config["model_config"]
    cm_embd_ext = AASISTModel(model_config).to(device)
    load_parameters(cm_embd_ext.state_dict(), args.aasist_weight)
    cm_embd_ext.eval()

    asv_embd_ext = ECAPA_TDNN(C=1024).to(device)
    load_parameters(asv_embd_ext.state_dict(), args.ecapa_weight)
    asv_embd_ext.eval()
    
    for set_name in SET_PARTITION:
        save_embeddings(set_name, cm_embd_ext, asv_embd_ext, device)
    
    print("\nHoàn tất việc trích xuất embedding cho tất cả các file.")

if __name__ == "__main__":
    main()