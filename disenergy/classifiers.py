"""Lightweight topological classifiers (RBF-SVM, Random Forest, 2-layer MLP)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn


class TopologicalMLP(nn.Module):
    """Lightweight 2-layer MLP over cubical persistent homology feature vectors."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_tabular_classifier(kind: str = "rbf_svm", seed: int = 42) -> Pipeline:
    """Create a deterministic sklearn classifier pipeline for topological descriptors."""
    k = kind.lower()
    if k == "rbf_svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)),
            ]
        )
    if k == "random_forest":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=-1)),
            ]
        )
    raise ValueError(f"Unknown tabular classifier '{kind}'. Choose 'rbf_svm' or 'random_forest'.")


def train_topological_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[TopologicalMLP, StandardScaler]:
    """Train the lightweight 2-layer TopologicalMLP deterministically."""
    torch.manual_seed(seed)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train).astype(np.float32)

    dev = torch.device(device)
    model = TopologicalMLP(in_dim=x_scaled.shape[1], num_classes=num_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    classes, counts = np.unique(y_train, return_counts=True)
    weights = np.ones(num_classes, dtype=np.float32)
    for c, cnt in zip(classes, counts):
        weights[int(c)] = float(len(y_train)) / (float(num_classes) * float(cnt))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=dev))

    x_t = torch.from_numpy(x_scaled).to(dev)
    y_t = torch.from_numpy(np.asarray(y_train, dtype=np.int64)).to(dev)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x_t)
        loss = criterion(logits, y_t)
        loss.backward()
        opt.step()

    model.eval()
    return model, scaler
