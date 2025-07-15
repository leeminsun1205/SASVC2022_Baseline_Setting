import os
import pickle as pk
import random
import sys
from tqdm import tqdm
from typing import Dict
from dataloaders.backend_fusion import collate_fn, SASV_SubmissionSet
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader



def str_to_bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    Copied from the python implementation distutils.utils.strtobool

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    >>> str_to_bool('YES')
    1
    >>> str_to_bool('FALSE')
    0
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError("invalid truth value {}".format(val))


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def keras_decay(step, decay=0.0001):
    """Learning rate decay in Keras-style"""
    return 1.0 / (1.0 + decay * step)


def set_seed(args):
    """
    set initial seed for reproduction
    """

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = args.cudnn_deterministic_toggle
        torch.backends.cudnn.benchmark = args.cudnn_benchmark_toggle


def set_init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        try:
            m.bias.data.fill_(0.0001)
        except:
            pass
    elif isinstance(m, nn.BatchNorm1d):
        pass
    else:
        try:
            torch.nn.init.kaiming_normal_(m.weight, a=0.01)
        except:
            pass

def load_parameters(trg_state, path):
    loaded_state = torch.load(path, map_location=lambda storage, loc: storage)
    for name, param in loaded_state.items():
        origname = name
        if name not in trg_state:
            name = name.replace("module.", "")
            name = name.replace("speaker_encoder.", "")
            if name not in trg_state:
                print("%s is not in the model."%origname)
                continue
        if trg_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, trg_state[name].size(), loaded_state[origname].size()))
            continue
        trg_state[name].copy_(param)


def find_gpus(nums=4, min_req_mem=None) -> str:
    """
    Allocates 'nums' GPUs that have the most free memory.
    Original source:
    https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/10

    :param nums: number of GPUs to find
    :param min_req_mem: required GPU memory (in MB)
    :return: string of GPU indices separated with comma
    """

    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_free_gpus')
    with open('tmp_free_gpus', 'r', encoding="utf-8") as lines_txt:
        frees = lines_txt.readlines()
        idx_freememory_pair = [ (idx,int(x.split()[2]))
                              for idx,x in enumerate(frees) ]
    idx_freememory_pair.sort(key=lambda my_tuple:my_tuple[1],reverse=True)
    using_gpus = [str(idx_memory_pair[0])
                    for idx_memory_pair in idx_freememory_pair[:nums] ]

    # return error signal if minimum required memory is given and
    # at least one GPU does not have sufficient memory
    if min_req_mem is not None and \
        int(idx_freememory_pair[nums][1]) < min_req_mem:

        return -1

    using_gpus =  ','.join(using_gpus)
    print('using GPU idx: #', using_gpus)
    return using_gpus


def get_spkdic(cm_meta: str) -> Dict:
    
    """
    return:
    {
        'S001': {
            'bonafide': ['LA_000001-F-0010001.wav', 'LA_000003-F-0010003.wav'],
            'spoof': []
        },
        'S002': {
            'bonafide': ['LA_000005-F-0010005.wav'],
            'spoof': ['LA_000002-F-0010002.wav']
        },
        'S003': {
            'bonafide': [],
            'spoof': ['LA_000004-F-0010004.wav']
        }
    }
    """
    
    l_cm_meta = open(cm_meta, "r").readlines()

    d_spk = {}

    for line in l_cm_meta:
        spk, filename, ans = line.strip().split(" ")
        
        if spk not in d_spk:
            d_spk[spk] = {}
            d_spk[spk]["bonafide"] = []
            d_spk[spk]["spoof"] = []

        if ans == "bonafide":
            d_spk[spk]["bonafide"].append(filename)
        elif ans == "spoof":
            d_spk[spk]["spoof"].append(filename)

    return d_spk


def generate_spk_meta(config) -> None:
    d_spk_train = get_spkdic(config.dirs.cm_trn_list)
    d_spk_dev = get_spkdic(config.dirs.cm_dev_list)
    d_spk_eval = get_spkdic(config.dirs.cm_eval_list)

    os.makedirs(config.dirs.spk_meta, exist_ok=True)

    with open(config.dirs.spk_meta + "spk_meta_trn.pk", "wb") as f:
        pk.dump(d_spk_train, f)
    with open(config.dirs.spk_meta + "spk_meta_dev.pk", "wb") as f:
        pk.dump(d_spk_dev, f)
    with open(config.dirs.spk_meta + "spk_meta_eval.pk", "wb") as f:
        pk.dump(d_spk_eval, f)

def get_unique_files_from_trial(trial_file: str) -> list:
    if not os.path.exists(trial_file):
        return []
    
    unique_files = set()
    with open(trial_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                if '/' in parts[0]: unique_files.add(parts[0])
                if '/' in parts[1]: unique_files.add(parts[1])
    return list(unique_files)

def generate_submission(system, trial_path: str, output_path: str):

    with open(trial_path, "r") as f:
        trials = [line.strip() for line in f if line.strip()]
    print("▶️ Số dòng trong trial:", len(trials))
    print("▶️ Ví dụ trial:", trials[:5])
    # Tạo dataset và dataloader
    dataset = SASV_SubmissionSet(trials, system.cm_embd_public_test, system.asv_embd_public_test)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        num_workers=system.config.loader.n_workers,
        collate_fn=collate_fn
    )

    # Đảm bảo hệ thống ở chế độ eval
    system.eval()
    results = []

    # Lấy thiết bị của model
    device = next(system.model.parameters()).device

    with torch.no_grad():
        for batch in tqdm(loader, desc="🔮 Đang tạo dự đoán"):
            if batch[0] is None:
                continue
            embd_asv_enrol, embd_asv_test, embd_cm_test, keys = batch

            # Đưa tất cả input về đúng device
            embd_asv_enrol = embd_asv_enrol.to(device)
            embd_asv_test = embd_asv_test.to(device)
            embd_cm_test = embd_cm_test.to(device)

            # Dự đoán
            pred = system.model(embd_asv_enrol, embd_asv_test, embd_cm_test)
            pred = torch.softmax(pred, dim=-1)[:, 1].detach().cpu().numpy()

            # Ghi kết quả
            for key, score in zip(keys, pred):
                parts = key.strip().split()
                if len(parts) != 2:
                    print(f"⚠️  Bỏ qua key không hợp lệ: '{key}'")
                    continue
                enr, tst = parts
                results.append(f"{enr}\t{tst}\t{score:.5f}")

    # Ghi file kết quả
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("enrollment_wav\ttest_wav\tscore\n")  # header
        f.write("\n".join(results) + "\n")

    print(f"✅ Submission saved to {output_path} with {len(results)} entries.")

