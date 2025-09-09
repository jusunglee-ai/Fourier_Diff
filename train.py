
# train.py
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
    save_checkpoint,
    save_image_tensor,
)
from modules.fft_utils import pad_to_multiple, unpad  # <-- 패딩/언패딩

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
    parser = argparse.ArgumentParser(description='Strict LFDM (paper-exact) — direct LFDMNetWithEMA trainer')
    parser.add_argument("--config", default='supervised.yml', type=str, help="Path to the config file")
    parser.add_argument('--resume', default='', type=str, help='Checkpoint to load and resume (optional)')
    parser.add_argument("--image_folder", default='results/', type=str, help="Where to save validation images")
    parser.add_argument('--seed', default=230, type=int, metavar='N', help='Random seed')
    args = parser.parse_args()

    # 현재 폴더에 yml이 있으면 아래 한 줄로 바꾸세요:
    # with open(args.config, "r") as f:
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    return args, new_config

@torch.no_grad()
def run_validation(model_dp, val_loader, device, out_dir, S=None):
    """
    검증 시 입력을 64의 배수로 패딩 → 샘플 → 언패딩 (Encoder 3down * UNet 3down = 2^6)
    """
    model_dp.eval()
    if hasattr(model_dp.module, "swap_to_ema"):
        model_dp.module.swap_to_ema()

    os.makedirs(out_dir, exist_ok=True)
    for i, batch in enumerate(val_loader):
        Ilow = batch[0].to(device)
        name = batch[-1][0] if isinstance(batch[-1], (list, tuple)) else f"{i:06d}.png"

        # *** 핵심: 64의 배수로 패딩 ***
        H, W = Ilow.shape[-2], Ilow.shape[-1]
        Ilow_pad, pad = pad_to_multiple(Ilow, mult=64, mode='reflect')

        pred = model_dp.module.sample(Ilow_pad, S=S)  # {"I_hat_low": ...}
        I_hat = pred["I_hat_low"]

        # 원래 크기로 복원
        if pad != (0, 0, 0, 0):
            I_hat = unpad(I_hat, pad)
        I_hat = I_hat[:, :, :H, :W]

        save_image_tensor(I_hat, os.path.join(out_dir, name))

    if hasattr(model_dp.module, "swap_to_raw"):
        model_dp.module.swap_to_raw()
    model_dp.train()

def main():
    args, config = parse_args_and_config()

    # device & seed
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)
    config.device = device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # dataset
    print("=> using dataset '{}'".format(config.data.train_dataset))
    DATASET = datasets.__dict__[config.data.type](config)
    train_loader, val_loader = DATASET.get_loaders()

    # build model
    cfg = DiffusionConfig(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        latent_ch=config.model.latent_ch,
        image_size=config.model.latent_size,  # 최소 기준(가이드)값. 실제 입력은 패딩으로 맞춤
        lambda1=getattr(config.training, "lambda1", 1.0),
        lambda2=getattr(config.training, "lambda2", 1.0),
    )
    model = LFDMNetWithEMA(cfg, use_ema=True).to(device)
    model_dp = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model_dp.parameters(), lr=config.training.lr)

    start_epoch, step = 0, 0

    # (옵션) resume
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model_dp.load_state_dict(ckpt["state_dict"], strict=True)
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("ema_helper") and model_dp.module.ema_helper is not None:
            model_dp.module.ema_helper.load_state_dict(ckpt["ema_helper"])
        start_epoch = ckpt.get("epoch", 0)
        step = ckpt.get("step", 0)
        print(f"=> Resumed from {args.resume} at epoch {start_epoch}, step {step}")

    # EMA 초기화
    if hasattr(model_dp.module, "start_ema"):
        model_dp.module.start_ema()

    # ---------------- Stage-1: AE pretrain ----------------
    n_epochs_stage1 = getattr(config.training, "n_epochs_stage1", 0)
    if n_epochs_stage1 > 0:
        print(f"[Stage-1] AE pretraining for {n_epochs_stage1} epochs ...")
        model_dp.train()
        for ep in range(start_epoch, n_epochs_stage1):
            for batch in train_loader:
                x = batch[0].to(device)  # (Ilow)만 써도 AE 충분
                out = model_dp.module.stage1(x)
                loss = out["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if hasattr(model_dp.module, "update_ema"):
                    model_dp.module.update_ema()
                step += 1

                if step % max(1, getattr(config.training, "log_freq", 50)) == 0:
                    print(f"[S1 step {step}] L_stage1:{loss.item():.5f}", flush=True)

    # encoder freeze (논문)
    for p in model_dp.module.encoder.parameters():
        p.requires_grad_(False)

    # ---------------- Stage-2: LFDM supervised ----------------
    print(f"[Stage-2] LFDM supervised training for {getattr(config.training, 'n_epochs_stage2', 0)} epochs ...")
    model_dp.train()
    for ep in range(getattr(config.training, "n_epochs_stage2", 0)):
        for batch in train_loader:
            if not (isinstance(batch, (tuple, list)) and len(batch) >= 2 and torch.is_tensor(batch[1])):
                raise RuntimeError("Stage-2 requires paired batches (Ilow, Ihigh). Check your dataset loader.")
            Ilow, Ihigh = batch[0].to(device), batch[1].to(device)

            # *** 핵심: 64의 배수로 패딩 (학습 시에도) ***
            Ilow_pad, pad_l = pad_to_multiple(Ilow,  mult=64, mode='reflect')
            Ihigh_pad, pad_h = pad_to_multiple(Ihigh, mult=64, mode='reflect')
            # 주의: 두 쌍이 동일 spatial이어야 하므로, 일반적으로 동일한 크기 데이터셋이면 pad_l==pad_h.
            # 혹시 다르면 더 큰 패딩으로 다시 맞춰주는 로직을 넣을 수도 있음(필요시 알려줘).

            out = model_dp.module.stage2(Ilow_pad, Ihigh_pad)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if hasattr(model_dp.module, "update_ema"):
                model_dp.module.update_ema()
            step += 1

            if step % max(1, getattr(config.training, "log_freq", 50)) == 0:
                print(
                    f"[S2 step {step}] "
                    f"L_total:{out['L_total']:.4f}  L_diff:{out['L_diff']:.4f}  "
                    f"L_simple:{out['L_simple']:.4f}  L_phase:{out['L_phase']:.4f}  "
                    f"L_con:{out['L_con']:.4f}  L_fre:{out['L_fre']:.4f}",
                    flush=True
                )

            # 검증 & 체크포인트
            if step % getattr(config.training, "validation_freq", 1000) == 0:
                S = getattr(config.diffusion, "num_sampling_timesteps", None)  # DDPM S-step (없으면 full-T)
                out_dir = os.path.join(args.image_folder, f"step_{step}")
                run_validation(model_dp, val_loader, device, out_dir, S=S)

                ckpt = {
                    "step": step,
                    "epoch": ep + 1,
                    "state_dict": model_dp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema_helper": model_dp.module.ema_helper.state_dict() if model_dp.module.ema_helper else None,
                    "params": vars(args),
                    "config": config.__dict__,
                }
                save_checkpoint(ckpt, filename=os.path.join(config.data.ckpt_dir, "model_latest.pth"))

if __name__ == "__main__":
    main()
