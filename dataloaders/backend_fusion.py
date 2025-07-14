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
    # Bỏ spk_model khỏi __init__
    def __init__(self, utt_list, cm_embd, asv_embd):
        self.utt_list = utt_list
        self.cm_embd = cm_embd
        self.asv_embd = asv_embd

    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, index):
        line = self.utt_list[index].strip()
        parts = line.split(" ")
        
        # Chỉ xử lý định dạng: enroll_path test_path label
        if len(parts) == 3 and '/' in parts[0]:
            enroll_path, test_path, ans = parts
        else:
            return None, None, None, None # Bỏ qua các định dạng khác

        # Lấy embedding của file enrollment trực tiếp từ asv_embd
        enroll_emb = self.asv_embd.get(enroll_path)
        asv_emb = self.asv_embd.get(test_path)
        cm_emb = self.cm_embd.get(test_path)

        # Kiểm tra xem có lấy được embedding không
        if enroll_emb is None or asv_emb is None or cm_emb is None:
            return None, None, None, None

        return enroll_emb, asv_emb, cm_emb, ans


def get_trnset(
    cm_embd_trn: Dict, asv_embd_trn: Dict, spk_meta_trn: Dict
) -> SASV_DevEvalset:
    return SASV_Trainset(
        cm_embd=cm_embd_trn, asv_embd=asv_embd_trn, spk_meta=spk_meta_trn
    )


def get_dev_evalset(
    utt_list: List, cm_embd: Dict, asv_embd: Dict
) -> SASV_DevEvalset:
    # Bỏ spk_model khỏi lời gọi hàm
    return SASV_DevEvalset(
        utt_list=utt_list, cm_embd=cm_embd, asv_embd=asv_embd
    )
    
def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch: return None, None, None, None
    return list(zip(*batch))
