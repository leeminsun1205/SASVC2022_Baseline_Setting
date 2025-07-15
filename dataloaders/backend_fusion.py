import random
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset


class SASV_Trainset(Dataset):
    def __init__(self, cm_embd, asv_embd, spk_meta):
        self.cm_embd = cm_embd
        self.asv_embd = asv_embd
        self.spk_meta = spk_meta
        self.spk_list = list(self.spk_meta.keys())

    def __len__(self):
        return len(self.cm_embd.keys())

    def __getitem__(self, index):
        ans_type = random.randint(0, 1)
        if ans_type == 1:  # target
            spk = random.choice(self.spk_list)
            while len(self.spk_meta[spk]["bonafide"]) < 2:
                spk = random.choice(self.spk_list)
            enr, tst = random.sample(self.spk_meta[spk]["bonafide"], 2)

        elif ans_type == 0:  # nontarget
            nontarget_type = random.randint(1, 2)
            if nontarget_type == 1:  # zero-effort nontarget
                spk, ze_spk = random.sample(self.spk_list, 2)
                enr = random.choice(self.spk_meta[spk]["bonafide"])
                tst = random.choice(self.spk_meta[ze_spk]["bonafide"])
            else:  # spoof nontarget
                spk = random.choice(self.spk_list)
                while len(self.spk_meta[spk]["spoof"]) == 0:
                    spk = random.choice(self.spk_list)
                enr = random.choice(self.spk_meta[spk]["bonafide"])
                tst = random.choice(self.spk_meta[spk]["spoof"])

        return self.asv_embd[enr], self.asv_embd[tst], self.cm_embd[tst], ans_type


class SASV_DevEvalset(Dataset):
    def __init__(self, utt_list, cm_embd, asv_embd):
        self.utt_list = utt_list
        self.cm_embd = cm_embd
        self.asv_embd = asv_embd

    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, index):
        line = self.utt_list[index].strip()
        parts = line.split(" ")
        
        if len(parts) == 3 and '/' in parts[0]:
            enroll_path, test_path, ans = parts
        else:
            return None
            
        enroll_emb = self.asv_embd.get(enroll_path)
        asv_emb = self.asv_embd.get(test_path)
        cm_emb = self.cm_embd.get(test_path)

        if enroll_emb is None or asv_emb is None or cm_emb is None:
            return None

        return enroll_emb, asv_emb, cm_emb, ans

class SASV_SubmissionSet(Dataset):
    def __init__(self, utt_list, cm_embd, asv_embd):
        self.utt_list = utt_list
        self.cm_embd = cm_embd
        self.asv_embd = asv_embd

    def __len__(self):
        return len(self.utt_list)
    
    def __getitem__(self, index):
        line = self.utt_list[index].strip()
        parts = line.split()
        if len(parts) != 2:
            return None

        enroll_path, test_path = parts
        enroll_emb = self.asv_embd.get(enroll_path)
        asv_emb = self.asv_embd.get(test_path)
        cm_emb = self.cm_embd.get(test_path)

        if enroll_emb is None or asv_emb is None or cm_emb is None:
            return None

        return enroll_emb, asv_emb, cm_emb, f"{enroll_path} {test_path}"

def get_trnset(cm_embd_trn: Dict, asv_embd_trn: Dict, spk_meta_trn: Dict) -> SASV_Trainset:
    return SASV_Trainset(
        cm_embd=cm_embd_trn, 
        asv_embd=asv_embd_trn, 
        spk_meta=spk_meta_trn
    )


def get_dev_evalset(utt_list: List, cm_embd: Dict, asv_embd: Dict) -> SASV_DevEvalset:
    return SASV_DevEvalset(
        utt_list=utt_list, 
        cm_embd=cm_embd, 
        asv_embd=asv_embd
    )

def collate_fn(batch):

    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None, None

    enroll_embs, asv_embs, cm_embs, keys = zip(*batch)
    
    enroll_embs = torch.from_numpy(np.stack(enroll_embs, axis=0))
    asv_embs = torch.from_numpy(np.stack(asv_embs, axis=0))
    cm_embs = torch.from_numpy(np.stack(cm_embs, axis=0))
    
    return enroll_embs, asv_embs, cm_embs, keys