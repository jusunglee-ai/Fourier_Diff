# datasets/lol.py
import os
from glob import glob
from typing import Tuple, List, Dict
from PIL import Image, ImageFile, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# ---------- helpers ----------
def _load_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    try:
        img = ImageOps.exif_transpose(img)  # 안전한 EXIF 보정
    except Exception:
        pass
    return img.convert("RGB") if img.mode != "RGB" else img

def _pair_by_name(low_dir: str, high_dir: str) -> List[Tuple[str, str]]:
    def list_images(d):
        return [p for p in glob(os.path.join(d, "*")) if os.path.splitext(p)[1].lower() in IMG_EXTS]
    low_files  = list_images(low_dir)
    high_files = list_images(high_dir)

    def to_map(files):
        m = {}
        for p in files:
            stem = os.path.splitext(os.path.basename(p))[0]
            m[stem] = p
        return m

    low_map  = to_map(low_files)
    high_map = to_map(high_files)

    pairs, missing = [], []
    for stem, lp in sorted(low_map.items()):
        hp = high_map.get(stem)
        if hp is None:
            missing.append(stem)
        else:
            pairs.append((lp, hp))

    if not pairs:
        raise RuntimeError(f"No paired images under:\n  low: {low_dir}\n  high:{high_dir}")
    if missing:
        print(f"[LOL] Warning: {len(missing)} highs missing (first 5): {missing[:5]}")
    return pairs

# ---------- paired transforms ----------
class PairCompose(T.Compose):
    def __call__(self, a, b):
        for t in self.transforms:
            a, b = t(a, b)
        return a, b

class PairToTensor01(T.ToTensor):
    def __call__(self, a, b):
        return F.to_tensor(a), F.to_tensor(b)

class PairScaleMinus1To1:
    def __call__(self, a, b):
        return a * 2.0 - 1.0, b * 2.0 - 1.0

class PairRandomHorizontalFlip(T.RandomHorizontalFlip):
    def __call__(self, a, b):
        if torch.rand(1).item() < self.p:
            return F.hflip(a), F.hflip(b)
        return a, b

class PairRandomCrop(T.RandomCrop):
    def __call__(self, a, b):
        # pad-if-needed 동일 적용
        if self.padding is not None:
            a = F.pad(a, self.padding, self.fill, self.padding_mode)
            b = F.pad(b, self.padding, self.fill, self.padding_mode)
        if self.pad_if_needed and a.size[0] < self.size[1]:
            pad_w = self.size[1] - a.size[0]
            a = F.pad(a, (pad_w, 0), self.fill, self.padding_mode)
            b = F.pad(b, (pad_w, 0), self.fill, self.padding_mode)
        if self.pad_if_needed and a.size[1] < self.size[0]:
            pad_h = self.size[0] - a.size[1]
            a = F.pad(a, (0, pad_h), self.fill, self.padding_mode)
            b = F.pad(b, (0, pad_h), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(a, self.size)
        return F.crop(a, i, j, h, w), F.crop(b, i, j, h, w)


# ---------- datasets ----------
class LOLDatasetTrain(Dataset):
    """
    root/train/low/*.png
    root/train/high/*.png
    -> return: ( cat([Ilow, Ihigh], dim=0) in [-1,1], filename )
    """
    def __init__(self, root: str, patch_size: int = 128):
        self.low_dir  = os.path.join(root, "train", "low")
        self.high_dir = os.path.join(root, "train", "high")
        assert os.path.isdir(self.low_dir) and os.path.isdir(self.high_dir)
        self.pairs = _pair_by_name(self.low_dir, self.high_dir)
        self.tf = PairCompose([
            PairRandomHorizontalFlip(p=0.5),
            PairRandomCrop(size=patch_size),
            PairToTensor01(),
            PairScaleMinus1To1(),
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        lp, hp = self.pairs[idx]
        Il, Ih = _load_rgb(lp), _load_rgb(hp)
        Il, Ih = self.tf(Il, Ih)                # [-1,1], (3,H,W) each
        x6 = torch.cat([Il, Ih], dim=0)         # (6,H,W)
        name = os.path.basename(lp)
        return x6, name


class LOLDatasetVal(Dataset):
    """
    root/<split>/low/*.png
    root/<split>/high/*.png
    -> return: ( cat([Ilow, Ihigh], dim=0) in [-1,1], filename )
    """
    def __init__(self, root: str, split: str = "test"):
        assert split in ("val", "test", "train")
        self.low_dir  = os.path.join(root, split, "low")
        self.high_dir = os.path.join(root, split, "high")
        assert os.path.isdir(self.low_dir) and os.path.isdir(self.high_dir)
        self.pairs = _pair_by_name(self.low_dir, self.high_dir)
        self.tf = PairCompose([
            PairToTensor01(),
            PairScaleMinus1To1(),
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        lp, hp = self.pairs[idx]
        Il, Ih = _load_rgb(lp), _load_rgb(hp)
        Il, Ih = self.tf(Il, Ih)                # [-1,1]
        x6 = torch.cat([Il, Ih], dim=0)         # (6,H,W)
        name = os.path.basename(lp)
        return x6, name


class lol:  # train.py에서 datasets.__dict__['lol'](config) 형태로 사용
    def __init__(self, config):
        self.cfg = config
        root = config.data.root
        ps   = getattr(config.data, "patch_size", 128)
        split_val = getattr(config.data, "val_split", "test")
        self.train_set = LOLDatasetTrain(root, patch_size=ps)
        self.val_set   = LOLDatasetVal(root, split=split_val)

        self.batch_size  = getattr(config.data, "batch_size", 4)
        self.num_workers = getattr(config.data, "num_workers", 4)

    def get_loaders(self):
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader
