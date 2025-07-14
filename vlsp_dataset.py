# vlsp_dataset.py
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path

def pad(x, max_len=64600):
    """Hàm đệm audio để đảm bảo tất cả các mẫu có cùng độ dài."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # Lặp lại audio cho đến khi đủ độ dài
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

class VLSPDataset(Dataset):
    """
    Lớp Dataset được thiết kế riêng để đọc dữ liệu âm thanh của cuộc thi VLSP.
    Nó nhận vào một danh sách các đường dẫn tương đối và một thư mục gốc.
    """
    def __init__(self, file_paths, base_dir):
        self.file_paths = file_paths
        self.base_dir = Path(base_dir)
        self.cut = 64600  # Lấy khoảng ~4 giây âm thanh (64600 mẫu)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # Lấy đường dẫn tương đối từ danh sách
        relative_path = self.file_paths[index]
        
        # Kết hợp thư mục gốc và đường dẫn tương đối để có đường dẫn đầy đủ
        full_path = self.base_dir / relative_path
        
        X, _ = sf.read(str(full_path))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        
        # Trả về tensor âm thanh và đường dẫn tương đối (dùng làm key)
        return x_inp, relative_path