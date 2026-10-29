"""Feed-forward MLP, explicit training loop and run caching (Parts 2, 3 and 5).

Autodiff (loss.backward()) computes the gradients, but the parameter update is
written out by hand in `sgd_momentum_step` so it is visible exactly how gradients
change the weights.
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ROOT

RUN_DIR = ROOT / "results" / "runs"
CKPT_DIR = ROOT / "results" / "checkpoints"

BASELINE = dict(hidden=[256, 128], dropout=0.0, lr=0.05, momentum=0.9, batch_size=128,
                epochs=60, weight_decay=0.0, seed=0)


# ----------------------------------------------------------------------------- model
class MLP(nn.Module):
    """784 -> hidden[0] -> ... -> hidden[-1] -> 10, ReLU after every hidden layer.

    forward(x) returns logits (unnormalised class scores); softmax is applied
    inside the cross-entropy loss. features(x) returns the activations of the
    final hidden layer (used as a learned representation in Part 5).
    """

    def __init__(self, hidden=(256, 128), dropout=0.0):
        super().__init__()
        layers, d_in = [], 784
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))  # active only in model.train() mode
            d_in = h
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(d_in, 10)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He/Kaiming initialisation keeps activation variance stable through ReLU layers
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def features(self, x):
        return self.hidden(x)

    def forward(self, x):
        return self.out(self.hidden(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------ gradient update
def sgd_momentum_step(model, velocity, lr, momentum, weight_decay):
    """One SGD-with-momentum update, written explicitly.

        g  = dL/dw  (+ weight_decay * w   for weight matrices: gradient of (wd/2)||w||^2)
        v <- momentum * v + g
        w <- w - lr * v
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            g = p.grad
            if weight_decay > 0 and p.dim() > 1:  # L2 on weights only, not biases
                g = g + weight_decay * p
            v = velocity[name]
            v.mul_(momentum).add_(g)
            p.sub_(lr * v)


# --------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, X, y, batch_size=2000):
    """Mean cross-entropy (data term only, no penalty) and accuracy, in eval mode
    (dropout off), so losses are comparable between regularised and unregularised runs."""
    model.eval()
    total_loss, correct = 0.0, 0
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        total_loss += F.cross_entropy(logits, y[i:i + batch_size], reduction="sum").item()
        correct += (logits.argmax(1) == y[i:i + batch_size]).sum().item()
    return total_loss / len(X), correct / len(X)


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return F.softmax(model(torch.as_tensor(X)), dim=1).numpy()


# ------------------------------------------------------------------ training loop
def train(cfg, X_tr, y_tr, X_val, y_val, verbose=False):
    """Train one MLP from scratch. Returns (model at best validation epoch, history dict).

    Checkpoint rule: keep the parameters from the epoch with the highest validation
    accuracy (ties -> earliest epoch). Training always runs for all epochs so the full
    learning curves are recorded.
    Sources of randomness (all fixed by cfg['seed']): weight initialisation, minibatch
    order and dropout masks. The data split is fixed separately (SPLIT_SEED).
    """
    torch.manual_seed(cfg["seed"])                      # -> initialisation and dropout masks
    shuffle_gen = torch.Generator().manual_seed(cfg["seed"])  # -> minibatch order
    model = MLP(cfg["hidden"], cfg["dropout"])
    velocity = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    X_tr, y_tr = torch.as_tensor(X_tr), torch.as_tensor(y_tr)
    X_val, y_val = torch.as_tensor(X_val), torch.as_tensor(y_val)
    n = len(X_tr)
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "running_loss"]}
    best = {"val_acc": -1.0, "epoch": 0, "state": None}
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        # ---- training pass over shuffled minibatches
        model.train()
        perm = torch.randperm(n, generator=shuffle_gen)
        running = 0.0
        for i in range(0, n, cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            logits = model(X_tr[idx])                   # forward pass
            loss = F.cross_entropy(logits, y_tr[idx])   # softmax + negative log-likelihood
            for p in model.parameters():
                p.grad = None                           # clear old gradients
            loss.backward()                             # backpropagation: dL/dw for every w
            sgd_momentum_step(model, velocity, cfg["lr"], cfg["momentum"], cfg["weight_decay"])
            running += loss.item() * len(idx)

        # ---- validation loop (and a clean re-evaluation of the training set)
        tr_loss, tr_acc = evaluate(model, X_tr, y_tr)
        va_loss, va_acc = evaluate(model, X_val, y_val)
        for k, v in zip(hist, [tr_loss, tr_acc, va_loss, va_acc, running / n]):
            hist[k].append(v)
        if va_acc > best["val_acc"]:
            best = {"val_acc": va_acc, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  ep {epoch:3d}  train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
        if not np.isfinite(tr_loss):                    # diverged: stop and record it
            print(f"  diverged at epoch {epoch}")
            break

    hist.update(best_epoch=best["epoch"], best_val_acc=best["val_acc"],
                n_params=count_params(model), train_time_s=time.time() - t0, config=cfg)
    model.load_state_dict(best["state"])
    return model, hist


def run(exp_id, cfg, X_tr, y_tr, X_val, y_val, retrain=False, verbose=False):
    """Train (or load from cache) one experiment. The cache is only reused if the
    stored configuration is identical to the requested one."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    hist_file, ckpt_file = RUN_DIR / f"{exp_id}.json", CKPT_DIR / f"{exp_id}.pt"
    if not retrain and hist_file.exists() and ckpt_file.exists():
        hist = json.loads(hist_file.read_text())
        if hist["config"] == cfg:
            model = MLP(cfg["hidden"], cfg["dropout"])
            model.load_state_dict(torch.load(ckpt_file))
            model.eval()
            return model, hist
    model, hist = train(cfg, X_tr, y_tr, X_val, y_val, verbose=verbose)
    hist_file.write_text(json.dumps(hist, indent=1))
    torch.save(model.state_dict(), ckpt_file)
    model.eval()
    return model, hist
