import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer
from models.resnet_s2m import S2MResNet
from data.ms2090a_interface import load_ms2090a_dat_file
import numpy as np
import os

class IQDataset(Dataset):
    def __init__(self, dat_dir, label_dict, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform
        for label_name, idx in label_dict.items():
            folder = os.path.join(dat_dir, label_name)
            for file in os.listdir(folder):
                if file.endswith(".dat"):
                    self.paths.append(os.path.join(folder, file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        iq = load_ms2090a_dat_file(self.paths[idx])
        if self.transform:
            iq = self.transform(iq)
        s2m = np.outer(iq.real, iq.imag).astype(np.float32)
        s2m = (s2m - s2m.mean()) / (s2m.std() + 1e-6)
        s2m = s2m[np.newaxis, :, :]
        label = self.labels[idx]
        return torch.tensor(s2m), torch.tensor(label)

class IQClassifier(LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = S2MResNet(num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# 用法示例
if __name__ == "__main__":
    label_dict = {"QPSK": 0, "BPSK": 1, "8PSK": 2}
    dataset = IQDataset("data/ms2090a", label_dict)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = IQClassifier(num_classes=len(label_dict))
    trainer = Trainer(max_epochs=10, accelerator="auto")
    trainer.fit(model, loader)
