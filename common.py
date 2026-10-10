"""Datasets, models and transforms shared by the training and evaluation scripts."""
import random, numpy as np, pandas as pd, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data_utils import key2file, load_table

# Segmentation augmentation actually used (reported verbatim in Table 4 of the revised manuscript)
SEG_TRAIN_TF = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.Affine(scale=(0.95, 1.05), translate_percent=(0.05, 0.05), rotate=0, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.2),
    ToTensorV2(),
])
DET_TRAIN_TF = A.Compose([A.HorizontalFlip(p=0.5), A.Rotate(limit=15, p=0.5),
                          A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), ToTensorV2()])
EVAL_TF = A.Compose([ToTensorV2()])      # deterministic: no augmentation at validation or test time


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class SliceDataset(Dataset):
    def __init__(self, table, transform, task="seg"):
        self.t = table.reset_index(drop=True); self.tf = transform; self.task = task

    def __len__(self): return len(self.t)

    def __getitem__(self, i):
        r = self.t.iloc[i]
        img = cv2.imread(r.image, cv2.IMREAD_GRAYSCALE).astype(np.float32)[..., None] / 255.0
        if self.task == "seg":
            msk = (cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
            o = self.tf(image=img, mask=msk)
            return o["image"], o["mask"][None].float() if o["mask"].ndim == 2 else o["mask"].float(), i
        o = self.tf(image=img)
        x = o["image"].repeat(3, 1, 1)
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return x, torch.tensor([float(r.has_liver)]), i


class UNetBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                   nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    def forward(self, x): return self.block(x)


class UNet(nn.Module):
    """Architecture as in the study code: encoder 64-128-256, bottleneck 512, three pooling levels,
    batch normalisation, transposed-convolution upsampling, 1x1 output + sigmoid (7.7 M parameters)."""
    def __init__(self, c_in=1, c_out=1):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = UNetBlock(c_in, 64), UNetBlock(64, 128), UNetBlock(128, 256)
        self.pool = nn.MaxPool2d(2); self.bottleneck = UNetBlock(256, 512)
        self.up3, self.dec3 = nn.ConvTranspose2d(512, 256, 2, 2), UNetBlock(512, 256)
        self.up2, self.dec2 = nn.ConvTranspose2d(256, 128, 2, 2), UNetBlock(256, 128)
        self.up1, self.dec1 = nn.ConvTranspose2d(128, 64, 2, 2), UNetBlock(128, 64)
        self.out_conv = nn.Conv2d(64, c_out, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1)); d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.out_conv(d1))


def resnet50_detector(pretrained=True):
    import torchvision
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = torchvision.models.resnet50(weights=w); m.fc = nn.Linear(m.fc.in_features, 1)
    return m          # logits; sigmoid applied in loss / at inference


def dice_loss(p, t, eps=1e-6):
    inter = (p * t).sum((2, 3)); s = p.sum((2, 3)) + t.sum((2, 3))
    return 1 - ((2 * inter + eps) / (s + eps)).mean()
