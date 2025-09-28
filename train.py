import os, argparse, yaml, glob, random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.ddm import DenoisingDiffusion
from datasets.dataset import lol as LOLDataset  # LOL-v1용
# utils.logging / utils.optimize 는 ddm 내부에서 사용


def set_seed(seed):
    if seed is None: return
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def to_ns(d):
    from types import SimpleNamespace
    def rec(x):
        if isinstance(x, dict): return SimpleNamespace(**{k: rec(v) for k, v in x.items()})
        if isinstance(x, list): return [rec(v) for v in x]
        return x
    return rec(d)


# top-level Args (로컬 클래스 pickle 이슈 회피)
class Args:
    pass


def _find_latest_stage2_ckpt(ckpt_dir: str) -> str:
    latest = os.path.join(ckpt_dir, "model_latest.pth.tar")
    if os.path.isfile(latest):
        return latest
    cands = glob.glob(os.path.join(ckpt_dir, "model_step*.pth.tar"))
    if not cands:
        return ""
    return max(cands, key=os.path.getmtime)


def train_stage2(cfg_dict):
    cfg = to_ns(cfg_dict)
    assert hasattr(cfg.data, "root"), "config.data.root 를 설정하세요 (LOL-v1 루트)"
    device = torch.device(cfg.device if hasattr(cfg, "device") else "cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device
    set_seed(getattr(cfg, "seed", None))

    # ddm이 args.resume / image_folder / mode 사용
    args = Args()
    # 1) yaml에 resume 주면 우선
    args.resume = getattr(cfg.training, "resume", "")
    # 2) 없으면 자동 탐색
    if not args.resume:
        auto = _find_latest_stage2_ckpt(cfg.data.ckpt_dir)
        if auto:
            print(f"[Stage2] auto-resume: {auto}")
            args.resume = auto
        else:
            print("[Stage2] no checkpoint found, start fresh")

    args.image_folder = cfg.data.image_folder
    args.mode = "training"

    # dataset 만들고 학습
    dataset = LOLDataset(cfg)
    trainer = DenoisingDiffusion(args, cfg)
    trainer.train(dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)  # configs/stage2.yml
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["data"]["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["data"]["image_folder"], exist_ok=True)

    train_stage2(cfg)


if __name__ == "__main__":
    main()
