"""One epoch of training, and prediction over arrays. No file reading or writing."""
import numpy as np
import torch
from torch.utils.data import DataLoader


def pick_device(name: str = "auto") -> torch.device:
    """``auto`` means the GPU when there is one, otherwise the CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_one_epoch(model, loader: DataLoader, loss_fn, optimizer, device) -> float:
    """Run one pass over the training loader. Returns the mean loss."""
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(y); n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predict_loader(model, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred)`` for every beat in the loader."""
    model.eval()
    trues, preds = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


@torch.no_grad()
def predict_array(model, x: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    """Class scores (n, num_classes) for an array of model inputs of shape (n, 1, L)."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch_size], dtype=np.float32)).to(device)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out)
