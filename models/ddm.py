import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import utils
from .unet import DiffusionUNet
from .autoencoder import FFTDecomposer,IFFT2d
from modules.Transformer_modules import LBM, LFFM


# -------------------------
# EMA helper
# -------------------------
class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, p in module.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, p in module.named_parameters():
            if p.requires_grad:
                self.shadow[name].data = (1. - self.mu) * p.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, p in module.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner = module.module
            m_copy = type(inner)(inner.config).to(inner.config.device)
            m_copy.load_state_dict(inner.state_dict())
            m_copy = nn.DataParallel(m_copy)
        else:
            m_copy = type(module)(module.config).to(module.config.device)
            m_copy.load_state_dict(module.state_dict())
        self.ema(m_copy)
        return m_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# -------------------------
# Diffusion schedule
# -------------------------
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x): return 1 / (np.exp(-x) + 1)
    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


# -------------------------
# Net (Stage1 encoder+FFT + LBM/LFFM + Diffusion U-Net)
# -------------------------
class Net(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device
        self.Unet = DiffusionUNet(config)
        self.ifft=IFFT2d(norm='ortho')
        # ---- Load Stage-1 (항상 로드; eval도 동일) ----

        if self.args.mode == 'training' :
            self.decom=self.load_stage1(FFTDecomposer(),"/home/jslee/LLIE/Fourier_Diff/ckpt/stage1/stage1_final.pth.tar")
        else:
            self.decom=FFTDecomposer()
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()

        self.num_timesteps = self.betas.shape[0]
        C = self.decom.base_channels
        self.LBM = LBM(channels=C).to(self.device)
        self.LFFM = LFFM(channels=C).to(self.device)




    @staticmethod
    def load_stage1(model, ckpt_path: str):
        ckpt = utils.logging.load_checkpoint(ckpt_path, 'cuda')
        model.load_state_dict(ckpt['model'], strict=True)
        return model


    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.config.diffusion.num_sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]


    def forward(self, inputs):
        data_dict = {}
        b = self.betas.to(inputs.device)

        if self.training:
            out = self.decom(inputs, pred_fea=None)

            low_A, low_P = out["low_A"], out["low_P"]
            high_A, high_P = out["high_A"], out["high_P"]
            low_fea=out["low_fea"]


            low_cond = torch.cat([torch.cos(low_P), torch.sin(low_P)], dim=1)


            t = torch.randint(low=0, high=self.num_timesteps, size=(low_cond.shape[0] // 2 + 1,)).to(
                self.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:low_cond.shape[0]].to(inputs.device)
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

            e = torch.randn_like(low_cond)

            high_input_norm=torch.cat([torch.cos(high_P), torch.sin(high_P)], dim=1)

            x = high_input_norm * a.sqrt() + e * (1.0 - a).sqrt()
            noise_output = self.Unet(torch.cat([low_cond, x], dim=1), t.float())

            pred_fea = self.sample_training(low_cond, b)

            cos_hat, sin_hat = torch.chunk(pred_fea, 2, dim=1)
            r = (cos_hat ** 2 + sin_hat ** 2).sqrt().clamp_min(1e-8)
            pred_fea= torch.atan2(sin_hat / r, cos_hat / r)
            lbm=self.LBM(low_A)

            ifft=self.ifft(lbm,pred_fea)
            lffm=self.LFFM(ifft,low_fea)
            output=self.decom(inputs,pred_fea=lffm)["pred_img"]

            data_dict["noise_output"] = noise_output
            data_dict["e"] = e
            data_dict["output"]=output
            data_dict["pred_fea"] = pred_fea
            data_dict["A_hat"]= lbm
            data_dict["F_hat_low"]=lffm
            data_dict["pred_img"]=output


        else:
            out = self.decom(inputs, pred_fea=None)
            low_phase = out["low_P"]
            low_A = out["low_A"]
            low_fea = out["low_fea"]
            outAmp = self.LBM(low_A)

            low_cond = torch.cat([torch.cos(low_phase), torch.sin(low_phase)], dim=1)
            pred_fea = self.sample_training(low_cond, b)

            cos_hat, sin_hat = torch.chunk(pred_fea, 2, dim=1)
            r = (cos_hat ** 2 + sin_hat ** 2).sqrt().clamp_min(1e-8)
            pred_fea = torch.atan2(sin_hat / r, cos_hat / r)

            latent_hat = self.ifft(outAmp, pred_fea)
            pred_LFFM = self.LFFM(latent_hat, low_fea)

            pred_x = self.decom(inputs, pred_fea=pred_LFFM)["pred_img"]
            data_dict["pred_img"] = pred_x

        return data_dict


# -------------------------
# Trainer
# -------------------------
class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config).to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))

        # freeze stage-1 (adapter to3는 학습 가능)
        for n, p in self.model.named_parameters():
            if "decom." in n and ("to3" not in n):
                p.requires_grad = False
            else:
                p.requires_grad = True

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.optimizer = utils.optimize.get_optimizer(
            self.config,
            filter(lambda p: p.requires_grad, self.model.parameters())
        )
        self.start_epoch, self.step = 0, 0

    # --------- resume/load ----------
    def load_ddm_ckpt(self, load_path, ema=False):
        ckpt = utils.logging.load_checkpoint(load_path, self.device)

        state_key = 'state_dict' if 'state_dict' in ckpt else 'model'
        self.model.load_state_dict(ckpt[state_key], strict=True)

        if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as e:
                print(f"[warn] optimizer state load skipped: {e}")

        if 'ema_helper' in ckpt and ckpt['ema_helper'] is not None:
            try:
                self.ema_helper.load_state_dict(ckpt['ema_helper'])
                if ema:
                    self.ema_helper.ema(self.model)
            except Exception as e:
                print(f"[warn] ema state load skipped: {e}")

        self.step = ckpt.get('step', 0)
        self.start_epoch = ckpt.get('epoch', 0)
        print(f"=> resume from: {load_path} | epoch={self.start_epoch} step={self.step}")

    # --------- loss ----------
    def noise_estimation_loss(self, out):
        pred_fea, high_phase = out["pred_fea"], out["high_P"]
        noise_out, e = out["noise_output"], out["e"]
        A_hat_low, A_high = out["A_hat"], out["high_A"]
        pred_img, high_img = out["pred_img"], out["I_high"]
        F_hat_low, F_high = out["F_hat_low"], out["F_high"]

        noise_loss = self.l2_loss(noise_out, e)
        phase_loss = self.l1_loss(pred_fea, high_phase)
        fre_loss = self.l2_loss(A_hat_low, A_high)

        cons_img = self.l2_loss(pred_img, high_img)
        cons_fea = self.l2_loss(F_hat_low, F_high)
        con_loss = cons_img + cons_fea

        lam1 = getattr(self.config.loss, "lambda_con", 1.0)
        lam2 = getattr(self.config.loss, "lambda_fre", 0.1)

        con_loss = lam1 * con_loss
        fre_loss = lam2 * fre_loss

        total = noise_loss + phase_loss + con_loss + fre_loss
        logs = {"noise": noise_loss, "phase": phase_loss, "con": con_loss, "fre": fre_loss}
        return total, logs

    # --------- validation saver ----------
    def sample_validation_patches(self, val_loader, step):
        """
        '현재 가중치'로 검증합니다. (EMA 사용하고 싶으면 config.training.val_use_ema=true)
        입력이 3채널(low) 또는 6채널(low+high) 모두 대응.
        """
        use_ema = getattr(self.config.training, "val_use_ema", False)
        save_root = os.path.join(self.args.image_folder, f"val_ps{self.config.data.patch_size}", str(step))
        os.makedirs(save_root, exist_ok=True)

        # --- (옵션) EMA로 검증 ---
        backup = None
        if use_ema:
            backup = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
            self.ema_helper.ema(self.model)

        self.model.eval()
        with torch.no_grad():
            print(f'Performing validation at step: {step}')
            for i, (x, y) in enumerate(val_loader):
                # x: [B,3,H,W] (low만) 또는 [B,6,H,W] (low+high)
                if x.shape[1] == 3:
                    low = x
                elif x.shape[1] == 6:
                    low = x[:, :3, :, :]
                else:
                    raise ValueError(f"unexpected C={x.shape[1]} (expected 3 or 6)")

                b, _, H, W = low.shape
                H64 = int(64 * np.ceil(H / 64.0))
                W64 = int(64 * np.ceil(W / 64.0))
                low_pad = F.pad(low, (0, W64 - W, 0, H64 - H), 'reflect')

                # 추론 입력은 [low, low] → 6채널
                x6 = torch.cat([low_pad, low_pad], dim=1)  # [B,6,H64,W64]

                out = self.model(x6.to(self.device))
                pred_x = out["pred_x"][:, :, :H, :W]  # [-1,1]

                # 저장: [0,1]
                pred_vis = utils.sampling.inverse_data_transform(pred_x.detach().cpu())
                utils.logging.save_image(pred_vis, os.path.join(save_root, f"{y[0]}"))

                # 입력/출력 비교 & diff(디버그)
                inp_vis = utils.sampling.inverse_data_transform(low.cpu())
                side = torch.cat([inp_vis, pred_vis], dim=-1)
                utils.logging.save_image(side, os.path.join(save_root, f"cmp_{y[0]}"))
                diff = (pred_vis - inp_vis).abs().mean(1, keepdim=True)
                diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
                utils.logging.save_image(diff, os.path.join(save_root, f"diff_{y[0]}"))

                # 수치 확인
                mae = (pred_vis - inp_vis).abs().mean().item()
                print(f"[val] {y[0]}  MAE={mae:.6f}")

        # --- EMA 사용했으면 원복 ---
        if use_ema and backup is not None:
            for n, p in self.model.named_parameters():
                if n in backup:
                    p.data.copy_(backup[n])

    # --------- train loop ----------
    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        # ----- robust resume -----
        resume_path = getattr(self.args, "resume", "")
        if resume_path and os.path.isfile(resume_path):
            try:
                self.load_ddm_ckpt(resume_path)
            except Exception as e:
                print(f"[Stage2] resume failed from {resume_path} ({e}); trying previous step ckpts...")
                ckpt_dir = self.config.data.ckpt_dir
                cands = sorted(
                    glob.glob(os.path.join(ckpt_dir, "model_step*.pth.tar")),
                    key=os.path.getmtime, reverse=True
                )
                loaded = False
                for p in cands:
                    try:
                        self.load_ddm_ckpt(p)
                        print(f"[Stage2] resumed from fallback: {p}")
                        loaded = True
                        break
                    except Exception as e2:
                        print(f"[Stage2] skip corrupt ckpt: {p} ({e2})")
                if not loaded:
                    print("[Stage2] no valid ckpt found; start fresh")
        else:
            print("[Stage2] fresh training (no resume)")

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch + 1)
            data_start = time.time()
            data_time = 0.0

            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                self.model.train()

                # freeze stage1 every iter (DP-safe)
                decom = self.model.module.decom if isinstance(self.model, nn.DataParallel) else self.model.decom
                decom.eval()
                for p in decom.parameters():
                    p.requires_grad = False

                self.step += 1
                x = x.to(self.device)
                out = self.model(x)
                total_loss, logs = self.noise_estimation_loss(out)

                data_time += time.time() - data_start

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.log_interval == 0:
                    print(f"step:{self.step}, "
                          f"noise:{logs['noise'].item():.5f} "
                          f"phase:{logs['phase'].item():.5f} "
                          f"con:{logs['con'].item():.5f} "
                          f"fre:{logs['fre'].item():.5f} "
                          f"time:{data_time / (i + 1):.5f}")

                if self.step % self.config.training.validation_freq == 0:
                    # --- 현재 가중치로 검증 ---
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                    # ----- save checkpoints: per-step + latest -----
                    state = {
                        'step': self.step,
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        # pickle 문제 방지: Args 자체 저장 X, dict로만 저장
                        'params': {
                            'resume': getattr(self.args, 'resume', ''),
                            'image_folder': getattr(self.args, 'image_folder', ''),
                            'mode': getattr(self.args, 'mode', ''),
                        },
                        'config': self.config,  # 필요시 문자열 경로만 저장하도록 바꿔도 OK
                    }
                    ckpt_dir = self.config.data.ckpt_dir
                    os.makedirs(ckpt_dir, exist_ok=True)

                    # 1) step 전용 (누적)
                    per_step = os.path.join(ckpt_dir, f"model_step{self.step:08d}")
                    utils.logging.save_checkpoint(state, filename=per_step)

                    # 2) latest (덮어쓰기)
                    latest = os.path.join(ckpt_dir, "model_latest")
                    utils.logging.save_checkpoint(state, filename=latest)

                    # 3) 오래된 step-ckpt 정리(선택)
                    keep_k = getattr(self.config.training, "keep_last_k", 10)
                    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "model_step*.pth.tar")),
                                   key=os.path.getmtime)
                    if len(ckpts) > keep_k:
                        for p in ckpts[:-keep_k]:
                            try:
                                os.remove(p)
                            except Exception:
                                pass
