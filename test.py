# test.py
import argparse
import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import datasets

from models.ddm import (
    LFDMNetWithEMA,
    DiffusionConfig,
    load_checkpoint,
    save_image_tensor,
)

def dict2namespace(config):
    ns = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(ns, key, new_value)
    return ns

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Strict LFDM Evaluation — direct LFDMNetWithEMA')
    parser.add_argument("--config", default='supervised.yml', type=str, help="Path to the config file")
    parser.add_argument('--ckpt', required=True, type=str, help='Checkpoint to load (model_latest.pth / model_final.pth)')
    parser.add_argument("--image_folder", default='results/test', type=str, help="Where to save test images")
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    return args, new_config

@torch.no_grad()
def main():
    args, config = parse_args_and_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config.device = device

    # dataset (val/test loader)
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()

    # build model
    cfg = DiffusionConfig(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        latent_ch=config.model.latent_ch,
        image_size=config.model.latent_size,
        lambda1=getattr(config.training, "lambda1", 1.0),
        lambda2=getattr(config.training, "lambda2", 1.0),
    )
    model = LFDMNetWithEMA(cfg, use_ema=True).to(device)
    model_dp = nn.DataParallel(model)

    # load checkpoint (+ EMA)
    ckpt = load_checkpoint(args.ckpt, device='cpu')
    model_dp.load_state_dict(ckpt["state_dict"], strict=True)
    if ckpt.get("ema_helper"):
        model_dp.module.ema_helper.load_state_dict(ckpt["ema_helper"])
        model_dp.module.swap_to_ema()

    model_dp.eval()
    outdir = args.image_folder
    os.makedirs(outdir, exist_ok=True)

    # 샘플링 스텝 설정 (있으면 S-step, 없으면 full-T)
    S = getattr(config.diffusion, "num_sampling_timesteps", None)

    for i, batch in enumerate(val_loader):
        # batch: (Ilow, name) — datasets/lol.py 기준
        Ilow = batch[0].to(device)
        name = batch[-1][0] if isinstance(batch[-1], (list, tuple)) else f"{i:06d}.png"

        pred = model_dp.module.sample(Ilow, S=S)  # DDPM S-step
        I_hat = pred["I_hat_low"]

        save_image_tensor(I_hat, os.path.join(outdir, name))

if __name__ == '__main__':
    main()
