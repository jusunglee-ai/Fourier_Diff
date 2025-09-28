# models/train_stage1.py
import os, time
import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from types import SimpleNamespace

from datasets.dataset import lol as LOLDataset
from models.autoencoder import FFTDecomposer
from utils.optimize import get_optimizer  # ★ utils.optimize 직접 임포트 (import utils 로 인한 경로 이슈 제거)


# ----------------- Stage-1 Trainer -----------------
class Stage1AE(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = config.device

        # ★ channels 키: config.model.channels, 없으면 64
        base_ch = getattr(config.model, "channels", getattr(config.model, "base_channels", 64))
        self.model = FFTDecomposer(channels=base_ch).to(self.device)

        # ★ DataParallel (1GPU여도 문제 없음)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))

        # Stage-1은 전체 학습
        for p in self.model.parameters():
            p.requires_grad = True

        self.l1 = nn.L1Loss()
        self.optimizer = get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        # ★ to3 존재 체크 (stage1 ckpt 호환: adapter_to3 / to3 둘 중 하나면 Ok)
        to3_ok = False
        mod = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(mod, "to3") and isinstance(mod.to3, nn.Conv2d):
            to3_ok = True
        elif hasattr(mod, "adapter_to3") and isinstance(mod.adapter_to3, nn.Conv2d):
            # 통일된 접근 함수를 붙여줌
            mod.to3 = mod.adapter_to3
            to3_ok = True
        if not to3_ok:
            raise RuntimeError("FFTDecomposer에 to3(또는 adapter_to3) 1x1 conv가 필요합니다. "
                               "forward(decoder) 전에 low/high_fea를 3ch로 투영합니다.")

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()  # (low+high) concat 입력: [B,6,H,W]

        # ★ ckpt_dir을 config에서 받음 (상대경로 꼬임 방지)
        ckpt_dir = self.config.data.ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print(f'[Stage1] epoch: {epoch}', flush=True)
            for i, (x, name) in enumerate(train_loader):
                self.model.train()
                self.step += 1
                x = x.to(self.device)  # [B,6,H,W]

                # --- encoder ---
                out = self.model(x, pred_fea=None)      # dict 예상: {"low_fea":..., "high_fea":...}
                low_fea, high_fea = out["low_fea"], out["high_fea"]

                # --- decoder 재구성 ---
                mod = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                low_code  = mod.to3(low_fea)            # [B,3,H/8,W/8] 가정
                high_code = mod.to3(high_fea)

                I_hat_low  = self.model(x, pred_fea=low_code)["pred_img"]   # [B,3,H,W]
                I_hat_high = self.model(x, pred_fea=high_code)["pred_img"]  # [B,3,H,W]

                low_img, high_img = x[:, :3, ...], x[:, 3:, ...]

                # --- AE L1 ---
                loss = self.l1(I_hat_low, low_img) + self.l1(I_hat_high, high_img)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                if self.step % max(1, getattr(self.config.training, "log_interval", 50)) == 0:
                    print(f"[Stage1] step:{self.step}  AE_L1:{loss.item():.5f}", flush=True)

            # --- 저장 ---
            save_path = os.path.join(ckpt_dir, "stage1_weight.pth.tar")
            torch.save({
                "model": self.model.state_dict(),
                "epoch": epoch,
                "step": self.step
            }, save_path)
            print(f"[Stage1] saved: {save_path}", flush=True)


# ----------------- entrypoint -----------------
def _load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _to_ns(d):
    def rec(x):
        if isinstance(x, dict): return SimpleNamespace(**{k: rec(v) for k, v in x.items()})
        if isinstance(x, list): return [rec(v) for v in x]
        return x
    return rec(d)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num_workers", type=int, default=None)  # 디버그 시 0 추천
    args = ap.parse_args()

    cfg_dict = _load_config(args.config)
    cfg = _to_ns(cfg_dict)

    # device 세팅
    device = torch.device(cfg.device if hasattr(cfg, "device") else "cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device

    # 워커 강제 (교착 시 0)
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    # 출력/체크포인트 폴더
    os.makedirs(cfg.data.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.data.image_folder, exist_ok=True)

    print("[Stage1] build dataset...", flush=True)
    dataset = LOLDataset(cfg)
    print(f"[Stage1] #train={len(dataset.train_set)}  #val={len(dataset.val_set)}", flush=True)

    print("[Stage1] build trainer...", flush=True)
    trainer = Stage1AE(args, cfg)

    print("[Stage1] start training...", flush=True)
    trainer.train(dataset)
    print("[Stage1] done.", flush=True)


if __name__ == "__main__":
    main()
