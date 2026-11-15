"""Two model families with hyperparameters fixed a priori (never tuned on DS2).

GBM   gradient-boosted trees on a decimated waveform + RR features.
CNN   small 1D residual network on the beat window, RR features fused late.
      Optional per-record BatchNorm re-estimation at test time (AdaBN), which
      uses the unlabeled beats of the test record only.
"""
from __future__ import annotations

import numpy as np

from .config import CLASSES
from .features import flat

K = len(CLASSES)


# ----------------------------------------------------------------------------- GBM
def fit_gbm(seq, tab, y, seed: int = 0, **override):
    import lightgbm as lgb

    params = dict(
        objective="multiclass", num_class=K,
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5, reg_lambda=1.0,
        class_weight="balanced", random_state=seed, n_jobs=2, verbose=-1,
    )
    params.update(override)
    model = lgb.LGBMClassifier(**params)
    model.fit(flat(seq, tab), y)
    return model


def predict_gbm(model, seq, tab):
    proba = model.predict_proba(flat(seq, tab))
    full = np.zeros((len(proba), K), dtype=np.float32)
    full[:, model.classes_.astype(int)] = proba  # a fold may lack a class
    return full


# ----------------------------------------------------------------------------- CNN
def _torch():
    import torch
    torch.set_num_threads(2)
    return torch


def _tab_prep(tab):
    # RR intervals are heavy-tailed (pauses of tens of seconds); compress them.
    return np.log1p(np.clip(tab, 0, 10)).astype(np.float32)


def build_cnn(in_ch: int, n_tab: int):
    torch = _torch()
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False)
            self.b1 = nn.BatchNorm1d(cout)
            self.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False)
            self.b2 = nn.BatchNorm1d(cout)
            self.skip = None
            if stride != 1 or cin != cout:
                self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False),
                                          nn.BatchNorm1d(cout))

        def forward(self, x):
            h = torch.relu(self.b1(self.c1(x)))
            h = self.b2(self.c2(h))
            return torch.relu(h + (x if self.skip is None else self.skip(x)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(in_ch, 16, 9, 2, 4, bias=False),
                                      nn.BatchNorm1d(16), nn.ReLU())
            self.body = nn.Sequential(Block(16, 16, 1), Block(16, 32, 2), Block(32, 64, 2))
            self.tab = nn.Sequential(nn.BatchNorm1d(n_tab), nn.Linear(n_tab, 32), nn.ReLU())
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64 + 32, K))

        def forward(self, x, t):
            h = self.body(self.stem(x)).mean(2)
            return self.head(torch.cat([h, self.tab(t)], 1))

    return Net()


def fit_cnn(seq, tab, y, seed: int = 0, epochs: int = 12, bs: int = 256, lr: float = 2e-3):
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.from_numpy(np.ascontiguousarray(seq))
    T = torch.from_numpy(_tab_prep(tab))
    Y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    net = build_cnn(seq.shape[1], tab.shape[1])
    # square-root inverse-frequency weights: full inverse weighting floods N
    # with false alarms when S is ~2% of beats.
    cnt = np.bincount(y, minlength=K).astype(float)
    w = np.where(cnt > 0, np.sqrt(cnt.max() / np.maximum(cnt, 1)), 0.0)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(Y) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    g = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Y), generator=g)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            xb = X[b]
            # mild augmentation: amplitude jitter and +-3 sample shift
            xb = xb * (1 + 0.1 * torch.randn(len(b), 1, 1))
            xb = torch.roll(xb, int(torch.randint(-3, 4, (1,))), dims=2)
            opt.zero_grad()
            loss_fn(net(xb, T[b]), Y[b]).backward()
            opt.step()
            sched.step()
    net.eval()
    return net


def predict_cnn(net, seq, tab, groups=None, adabn: bool = False, bs: int = 1024):
    """With adabn=True, BatchNorm statistics are re-estimated on each test
    record's own unlabeled beats before predicting that record."""
    torch = _torch()
    import copy

    X = torch.from_numpy(np.ascontiguousarray(seq))
    T = torch.from_numpy(_tab_prep(tab))
    out = np.zeros((len(X), K), dtype=np.float32)

    def run(model, idx):
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                out[b] = torch.softmax(model(X[b], T[b]), 1).numpy()

    if not adabn:
        run(net, np.arange(len(X)))
        return out

    for gid in np.unique(groups):
        idx = np.flatnonzero(groups == gid)
        m = copy.deepcopy(net)
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.reset_running_stats()
                mod.momentum = None  # cumulative average over the record
                mod.train()
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                if len(b) > 1:
                    m(X[b], T[b])
        m.eval()
        run(m, idx)
    return out
