# utils/logging.py
import torch, os

def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    import torchvision.utils as tvu
    tvu.save_image(img, file_directory)

def save_checkpoint(state, filename):
    # filename은 확장자 없이 들어올 수도 있으니 처리
    path = filename if filename.endswith('.pth.tar') else (filename + '.pth.tar')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic replace

def load_checkpoint(path, device=None):
    if device is None:
        return torch.load(path)
    else:
        return torch.load(path, map_location=device)
