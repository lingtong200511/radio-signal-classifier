import numpy as np
import os

def load_ms2090a_dat_file(file_path: str, sample_count=1024) -> np.ndarray:
    """
    从 MS2090A 导出的 .dat 文件中读取 IQ 信号（复数数组）
    :param file_path: 文件路径
    :param sample_count: 返回的样本点数
    :return: complex64 类型 numpy 数组
    """
    raw = np.fromfile(file_path, dtype=np.float32)
    assert len(raw) % 2 == 0, "数据长度必须为偶数"
    iq = raw[::2] + 1j * raw[1::2]
    return iq[:sample_count]
