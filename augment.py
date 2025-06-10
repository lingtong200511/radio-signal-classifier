import numpy as np

def iq_augment(iq: np.ndarray) -> np.ndarray:
    """
    基本的 IQ 数据增强：加噪、相位旋转、幅度缩放
    """
    noise = np.random.normal(0, 0.01, iq.shape)
    phase_shift = np.exp(1j * np.random.uniform(-np.pi, np.pi))
    scale = np.random.uniform(0.9, 1.1)
    return scale * (iq + noise) * phase_shift
