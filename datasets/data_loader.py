# datasets/lol.py
import os
from glob import glob
from typing import Tuple, List, Dict
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def _to_tensor_neg1_1():
    # [0,1] -> [-1,1]
    return T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x * 2.0 - 1.0)
    ])

def _load_rgb(path: str) -> Image.Image:
    # 안전한 로드 + EXIF 방향 보정 + RGBA -> RGB
    img = Image.open(path)
    try:
        img = Image.ImageOps.exif_transpose(img)  # type: ignore[attr-defined]
    except Exception:
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def _pair_by_name(low_dir: str, high_dir: str) -> List[Tuple[str, str]]:
    """low/high 파일을 '파일명(확장자 제외)' 기준으로 1:1 매칭."""
    low_files = [f for f in glob(os.path.join(low_dir, "*")) if os.path.splitext(f)[1].lower() in IMG_EXTS]
    high_files = [f for f in glob(os.path.join(high_dir, "*")) if os.path.splitext(f)[1].lower() in IMG_EXTS]

    def stem_map(files: List[str]) -> Dict[str, str]:
        m = {}
        for p in files:
            stem = os.path.splitext(os.path.basename(p))[0]
            m[stem] = p
        return m

    low_map = stem_map(low_files)
    high_map = stem_map(high_files)

    pairs: List[Tuple[str, str]] = []
    missing = []
    for stem, lp in sorted(low_map.items()):
        hp = high_map.get(stem, None)
        if hp is None:
            missing.append(stem)
        else:
            pairs.append((lp, hp))

    if not pairs:
        raise RuntimeError(f"No paired images between:\n  {low_dir}\n  {high_dir}")
    if missing:
        print(f"[LOL] Warning: {len(missing)} high images missing for lows (first 5): {missing[:5]}")

    return pairs

class LOLDatasetTrain(Dataset):
    """LOL-v1 Train: (Ilow, Ihigh) -> both in [-1,1]; random patch crop."""
    def __init__(self, root: str, patch_size: int = 128):
        """
        기대 구조:
          root/train/low/*.png
          root/train/high/*.png
        파일명 동일(확장자 무시)로 페어링.
        """
        self.low_dir = os.path.join(root, "train", "low")
        self.high_dir = os.path.join(root, "train", "high")
        assert os.path.isdir(self.low_dir) and os.path.isdir(self.high_dir), \
            f"Invalid train dirs under {root}"

        self.pairs = _pair_by_name(self.low_dir, self.high_dir)
        self.to_tensor = _to_tensor_neg1_1()
        self.patch_size = int(patch_size)

    def __len__(self):
        return len(self.pairs)

    def _random_crop(self, ilow: torch.Tensor, ihigh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ilow/ihigh: (C,H,W) in [-1,1]
        _, H, W = ilow.shape
        ps = self.patch_size

        if H < ps or W < ps:
            # 작은 이미지는 리사이즈(쌍방 동일)
            resize = T.Resize((max(ps, H), max(ps, W)), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
            # ToTensor 이전 스케일이 [0,1]이므로 역변환 후 ToTensor 경로와 동일하게 맞춰줌
            il = (ilow + 1) / 2.0
            ih = (ihigh + 1) / 2.0
            il = resize(T.ToPILImage()(il))
            ih = resize(T.ToPILImage()(ih))
            ilow = self.to_tensor(il)
            ihigh = self.to_tensor(ih)
            _, H, W = ilow.shape

        top = torch.randint(0, H - ps + 1, (1,)).item()
        left = torch.randint(0, W - ps + 1, (1,)).item()
        ilow = ilow[:, top:top+ps, left:left+ps]
        ihigh = ihigh[:, top:top+ps, left:left+ps]
        return ilow, ihigh

    def __getitem__(self, idx):
        low_path, high_path = self.pairs[idx]
        Il = _load_rgb(low_path)
        Ih = _load_rgb(high_path)

        Il = self.to_tensor(Il)   # [-1,1]
        Ih = self.to_tensor(Ih)   # [-1,1]

        Il, Ih = self._random_crop(Il, Ih)
        return Il, Ih  # (Ilow, Ihigh)

class LOLDatasetVal(Dataset):
    """LOL-v1 Test/Val: (Ilow, name) -> Ilow in [-1,1], full-size."""
    def __init__(self, root: str, split: str = "test"):
        """
        기대 구조:
          root/<split>/low/*.png
          root/<split>/high/*.png  (optional; 여기서는 사용 안 함)
        """
        assert split in ("val", "test", "train")
        self.low_dir = os.path.join(root, split, "low")
        assert os.path.isdir(self.low_dir), f"Invalid {split}/low dir under {root}"

        files = glob(os.path.join(self.low_dir, "*"))
        self.files = sorted([f for f in files if os.path.splitext(f)[1].lower() in IMG_EXTS])
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {self.low_dir}")

        self.to_tensor = _to_tensor_neg1_1()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        name = os.path.basename(path)
        Il = _load_rgb(path)
        Il = self.to_tensor(Il)   # [-1,1]
        return Il, name

class lol:
    """
    train.py에서 사용:
      DATASET = datasets.__dict__[config.data.type](config)
      train_loader, val_loader = DATASET.get_loaders()
    """
    def __init__(self, config):
        self.cfg = config
        root = config.data.root
        ps = getattr(config.data, "patch_size", 128)
        val_split = getattr(config.data, "val_split", "test")

        self.train_set = LOLDatasetTrain(root, patch_size=ps)
        self.val_set   = LOLDatasetVal(root, split=val_split)

        self.batch_size = getattr(config.data, "batch_size", 4)
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
