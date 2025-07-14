import argparse
import json
import os
import pickle as pk
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vlsp_dataset import VLSPDataset

from aasist.models.AASIST import Model as AASISTModel
from ECAPATDNN.model import ECAPA_TDNN
from utils import load_parameters

# list of dataset partitions
SET_PARTITION = ["trn", "dev", "eval"]

# list of countermeasure(CM) protocols
SET_CM_PROTOCOL = {
    "trn": "protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "protocols/ASVspoof2019.LA.cm.eval.trl.txt",
}

BASE_DIR = "/kaggle/input/vlsp-vsasv-datasets/vlsp2025/"

def save_embeddings(set_name, cm_embd_ext, asv_embd_ext, device):
    """Trích xuất và lưu ASV và CM embeddings."""
    meta_lines = open(SET_CM_PROTOCOL[set_name], "r").readlines()
    
    # Lấy danh sách đường dẫn tương đối từ tệp protocol
    # Định dạng: speaker_id file_path label
    utt_list = [line.strip().split(" ")[1] for line in meta_lines]

    dataset = VLSPDataset(file_paths=utt_list, base_dir=BASE_DIR)
    loader = DataLoader(dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True)

    cm_emb_dic = {}
    asv_emb_dic = {}

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
    with open(f"embeddings/cm_embd_{set_name}.pk", "wb") as f:
        pk.dump(cm_emb_dic, f)
    with open(f"embeddings/asv_embd_{set_name}.pk", "wb") as f:
        pk.dump(asv_emb_dic, f)


def save_models(set_name, asv_embd_ext, device):
    """
    Tạo và lưu speaker model bằng cách lấy trung bình embedding của các
    phát biểu 'bonafide' từ tập huấn luyện.
    """
    # Chúng ta sẽ sử dụng tệp train_split để tạo model cho TẤT CẢ các speaker
    # Điều này đảm bảo tính nhất quán cho cả tập dev và eval.
    meta_lines = open(SET_CM_PROTOCOL["trn"], "r").readlines()
    
    spk_utt_dic = {}
    # Đọc tất cả các phát biểu bonafide cho mỗi speaker
    for line in meta_lines:
        spk, utt, label = line.strip().split(" ")
        if label == "bonafide":
            if spk not in spk_utt_dic:
                spk_utt_dic[spk] = []
            spk_utt_dic[spk].append(utt)

    # Trích xuất embedding cho từng phát biểu
    all_utts = [utt for utts in spk_utt_dic.values() for utt in utts]
    
    dataset = VLSPDataset(file_paths=all_utts, base_dir=BASE_DIR)
    loader = DataLoader(dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True)
    
    utt_emb_dic = {}
    print(f"Trích xuất embedding để tạo speaker models...")
    for batch_x, key in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_asv_emb = asv_embd_ext(batch_x, aug=False).detach().cpu().numpy()
        for k, asv_emb in zip(key, batch_asv_emb):
            utt_emb_dic[k] = asv_emb

    # Tính toán speaker model bằng cách lấy trung bình
    spk_model_dic = {}
    for spk, utts in spk_utt_dic.items():
        embs = [utt_emb_dic[utt] for utt in utts if utt in utt_emb_dic]
        if embs:
            spk_model_dic[spk] = np.mean(embs, axis=0)

    # Lưu speaker model cho tập dev và eval (dùng chung một file)
    print(f"Lưu speaker models cho tập {set_name}...")
    with open(f"embeddings/spk_model_{set_name}.pk", "wb") as f:
        pk.dump(spk_model_dic, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-aasist_config", type=str, default="./aasist/config/AASIST.conf"
    )
    parser.add_argument(
        "-aasist_weight", type=str, default="./aasist/models/weights/AASIST.pth"
    )
    parser.add_argument(
        "-ecapa_weight", type=str, default="./ECAPATDNN/exps/pretrain.model"
    )

    return parser.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    # Tải mô hình CM (AASIST)
    with open(args.aasist_config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    cm_embd_ext = AASISTModel(model_config).to(device)
    load_parameters(cm_embd_ext.state_dict(), args.aasist_weight)
    cm_embd_ext.eval()

    # Tải mô hình ASV (ECAPA-TDNN)
    asv_embd_ext = ECAPA_TDNN(C=1024)
    load_parameters(asv_embd_ext.state_dict(), args.ecapa_weight)
    asv_embd_ext.eval()
    
    # Trích xuất embeddings cho các tập trn, dev, eval
    for set_name in SET_PARTITION:
        save_embeddings(set_name, cm_embd_ext, asv_embd_ext, device)

    # Tạo và lưu speaker models cho dev và eval
    # Chúng ta chỉ cần chạy một lần và lưu cho cả hai
    save_models("dev", asv_embd_ext, device)
    save_models("eval", asv_embd_ext, device)


if __name__ == "__main__":
    main()

