# trainer_strict_lfdm.py
import os, time, torch, torch.nn.functional as F
from torch.utils.data import DataLoader

# 1) 네가 이미 캔버스/리포에 둔 모델 오케스트레이터 (EMA 포함, DDPM full-T 샘플러)
from models.ddm import build_lfdm_with_ema, DiffusionConfig

class TrainerLFDM:
    def __init__(self, args, config, dataset):
        self.args = args
        self.config = config
        self.device = config.device
        self.dataset = dataset

        # 모델 빌드 (네 모듈들만 import해서 연결)
        self.model = build_lfdm_with_ema(
            DiffusionConfig(
                latent_ch=config.model.latent_ch,
                image_size=config.model.latent_size,
                beta_schedule=config.diffusion.beta_schedule,
                beta_start=config.diffusion.beta_start,
                beta_end=config.diffusion.beta_end,
                num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
                num_sampling_timesteps=config.diffusion.num_diffusion_timesteps  # STRICT: full T
            ),
            use_ema=True
        ).to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=config.training.lr)
        self.start_epoch, self.step = 0, 0

        # 로더 (기존 인터페이스 유지)
        self.train_loader, self.val_loader = dataset.get_loaders()

        # EMA 초기화
        self.model.start_ema()

    # ---------- Stage-1: Autoencoder pretrain ----------
    def train_stage1(self, epochs):
        self.model.train()
        for ep in range(epochs):
            for I, _ in self.train_loader:
                I = I.to(self.device)
                out = self.model.stage1(I)                # L = ||I - D(E(I))||^2
                loss = out["loss"]

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.model.update_ema()

    # ---------- Stage-2: Encoder freeze + LFDM ----------
    def train_stage2(self, epochs):
        # 인코더 고정 (논문)
        for p in self.model.encoder.parameters():
            p.requires_grad_(False)

        self.model.train()
        for ep in range(epochs):
            t0 = time.time()
            for i, (Ilow, Ihigh) in enumerate(self.train_loader):
                Ilow, Ihigh = Ilow.to(self.device), Ihigh.to(self.device)

                out = self.model.stage2(Ilow, Ihigh)      # L_total = L_diff + λ1 L_con + λ2 L_fre
                loss = out["loss"]

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.model.update_ema()
                self.step += 1

                if self.step % self.config.training.log_freq == 0:
                    print(f"[{self.step}] "
                          f"L_total:{out['L_total']:.4f}  "
                          f"L_diff:{out['L_diff']:.4f}  "
                          f"L_simple:{out['L_simple']:.4f}  "
                          f"L_phase:{out['L_phase']:.4f}  "
                          f"L_con:{out['L_con']:.4f}  "
                          f"L_fre:{out['L_fre']:.4f}  "
                          f"time:{(time.time()-t0):.2f}s"); t0 = time.time()

                if self.step % self.config.training.val_freq == 0:
                    self.validate_and_save(self.step)

    # ---------- Validation: strict DDPM full-T ----------
    @torch.no_grad()
    def validate_and_save(self, step):
        self.model.eval(); self.model.swap_to_ema()
        os.makedirs(self.args.image_folder, exist_ok=True)

        for Ilow, names in self.val_loader:
            Ilow = Ilow.to(self.device)
            # (필요하면 패딩 → 샘플 → 크롭. pad 헬퍼는 fft_utils에 있음)
            pred = self.model.sample(Ilow)                # DDPM full-T
            I_hat = pred["I_hat_low"]
            # TODO: 너의 utils.logging.save_image와 동일한 저장 API로 교체
            # utils.logging.save_image(I_hat, os.path.join(self.args.image_folder, f"{names[0]}"))

        # ckpt 저장 (EMA state도 함께)
        ckpt = {
            "step": step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "ema_helper": self.model.ema_helper.state_dict() if self.model.ema_helper else None,
            "params": self.args,
            "config": self.config,
        }
        # utils.logging.save_checkpoint(ckpt, filename=os.path.join(self.config.data.ckpt_dir, "model_latest"))
        self.model.swap_to_raw()
        self.model.train()
