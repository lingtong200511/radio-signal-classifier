import time
import torch
import numpy as np
from models.resnet_s2m import S2MResNet
from data.ms2090a_interface import load_ms2090a_dat_file

model = S2MResNet(num_classes=11)
model.load_state_dict(torch.load("model.ckpt", map_location="cpu"))
model.eval()

def classify(dat_file):
    iq = load_ms2090a_dat_file(dat_file)
    s2m = np.outer(iq.real, iq.imag).astype(np.float32)
    s2m = (s2m - s2m.mean()) / (s2m.std() + 1e-6)
    s2m = torch.tensor(s2m[np.newaxis, np.newaxis, :, :])
    with torch.no_grad():
        pred = model(s2m).argmax(dim=1).item()
    return pred

if __name__ == "__main__":
    while True:
        dat_path = input("输入最新采集到的 .dat 文件路径：")
        pred = classify(dat_path)
        print(f"预测调制类型类别编号：{pred}")
        time.sleep(1)
