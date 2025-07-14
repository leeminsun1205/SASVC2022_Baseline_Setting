import random
from typing import Dict, List

from torch.utils.data import Dataset


class SASV_Trainset(Dataset):
    def __init__(self, cm_embd, asv_embd, spk_meta):
        self.cm_embd = cm_embd
        self.asv_embd = asv_embd
        self.spk_meta = spk_meta
        # Chuyển đổi keys thành list một lần duy nhất ở đây để tối ưu
        self.spk_list = list(self.spk_meta.keys())

    def __len__(self):
        return len(self.cm_embd.keys())

    def __getitem__(self, index):

        ans_type = random.randint(0, 1)
        if ans_type == 1:  # target
            spk = random.choice(self.spk_list)
            # Đảm bảo có đủ 2 mẫu bonafide để lấy
            while len(self.spk_meta[spk]["bonafide"]) < 2:
                spk = random.choice(self.spk_list)
            
            enr, tst = random.sample(self.spk_meta[spk]["bonafide"], 2)

        elif ans_type == 0:  # nontarget
            nontarget_type = random.randint(1, 2)

            if nontarget_type == 1:  # zero-effort nontarget
                # SỬA Ở ĐÂY: Chuyển self.spk_meta.keys() thành list
                spk, ze_spk = random.sample(self.spk_list, 2)
                enr = random.choice(self.spk_meta[spk]["bonafide"])
                tst = random.choice(self.spk_meta[ze_spk]["bonafide"])

            if nontarget_type == 2:  # spoof nontarget
                spk = random.choice(self.spk_list)
                # Xử lý trường hợp người nói không có mẫu spoof
                while len(self.spk_meta[spk]["spoof"]) == 0:
                    spk = random.choice(self.spk_list)
                
                enr = random.choice(self.spk_meta[spk]["bonafide"])
                tst = random.choice(self.spk_meta[spk]["spoof"])

        return self.asv_embd[enr], self.asv_embd[tst], self.cm_embd[tst], ans_type


class SASV_DevEvalset(Dataset):
    def __init__(self, utt_list, spk_model, asv_embd, cm_embd):
        self.utt_list = utt_list
        self.spk_model = spk_model
        self.asv_embd = asv_embd
        self.cm_embd = cm_embd

    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, index):
        line = self.utt_list[index]
        spkmd, key, ans = line.strip().split(" ")

        return self.spk_model[spkmd], self.asv_embd[key], self.cm_embd[key], ans


def get_trnset(
    cm_embd_trn: Dict, asv_embd_trn: Dict, spk_meta_trn: Dict
) -> SASV_DevEvalset:
    return SASV_Trainset(
        cm_embd=cm_embd_trn, asv_embd=asv_embd_trn, spk_meta=spk_meta_trn
    )


def get_dev_evalset(
    utt_list: List, cm_embd: Dict, asv_embd: Dict, spk_model: Dict
) -> SASV_DevEvalset:
    return SASV_DevEvalset(
        utt_list=utt_list, cm_embd=cm_embd, asv_embd=asv_embd, spk_model=spk_model
    )
