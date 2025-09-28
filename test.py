import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import logging as Ulog
from datasets.dataset import lol as LOLDataset
from models.ddm import DenoisingDiffusion

def dict2ns(d):
    from types import SimpleNamespace
    def rec(x):
        if isinstance(x, dict):
            return SimpleNamespace(**{k: rec(v) for k, v in x.items()})
        elif isinstance(x, list):
            return [rec(v) for v in x]
        else:
            return x
    return rec(d)

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="/home/jslee/LLIE/Fourier_Diff/configs/stage2.yml",
                    help="stage2 config (must include training.stage1_ckpt)")
    ap.add_argument("--ckpt", type=str, default="/home/jslee/LLIE/Fourier_Diff/.ckpt/stage2/model_latest.pth.tar",
                    help="stage2 checkpoint (.pth.tar)")
    ap.add_argument("--image_folder", type=str, default="results/test",
                    help="save dir")
    ap.add_argument("--ema", action="store_true", default=False,
                    help="apply EMA weights for inference")
    args = ap.parse_args()

    cfg_dict = load_cfg(args.config)
    cfg = dict2ns(cfg_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device

    # 데이터셋: (x, name) — x는 [B,3,H,W] 또는 [B,6,H,W]
    DATASET = LOLDataset(cfg)
    _, val_loader = DATASET.get_loaders()

    # DDM wrapper (evaluation 모드)
    class Args: pass
    ddm_args = Args()
    ddm_args.resume = args.ckpt
    ddm_args.image_folder = args.image_folder
    ddm_args.mode = "evaluation"

    diffusion = DenoisingDiffusion(ddm_args, cfg)
    if os.path.isfile(args.ckpt):
        diffusion.load_ddm_ckpt(args.ckpt, ema=args.ema)
    diffusion.model.eval()

    os.makedirs(args.image_folder, exist_ok=True)

    for i, (x, name) in enumerate(val_loader):
        # x: [B,3,H,W] (low) 또는 [B,6,H,W] (low+high)
        if x.shape[1] == 3:
            low = x
        elif x.shape[1] == 6:
            low = x[:, :3]
        else:
            raise ValueError(f"unexpected C={x.shape[1]} (expected 3 or 6)")

        low = low.to(device)
        b, _, h, w = low.shape

        # 64 배수 패딩
        H64 = int(64 * torch.ceil(torch.tensor(h/64.0)).item())
        W64 = int(64 * torch.ceil(torch.tensor(w/64.0)).item())
        low_pad = F.pad(low, (0, W64 - w, 0, H64 - h), mode='reflect')

        # 추론 입력은 [low, low] → 6채널
        x6 = torch.cat([low_pad, low_pad], dim=1)

        out = diffusion.model(x6)
        pred = out["pred_x"][:, :, :h, :w]

        pred_01 = (pred.clamp(-1, 1) + 1) / 2.0
        Ulog.save_image(pred_01.cpu(), os.path.join(args.image_folder, name[0]))

        if (i+1) % 20 == 0:
            print(f"[test] {i+1}/{len(val_loader)} saved: {name[0]}")

if __name__ == "__main__":
    main()
