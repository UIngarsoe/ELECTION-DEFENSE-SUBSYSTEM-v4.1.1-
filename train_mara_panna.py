#!/usr/bin/env python3
"""
train_mara_panna.py
Training harness for MāraGenerator (Generator) vs PaññāDiscriminator (Discriminator)
Includes Baydin self-test, checkpointing, logging, and a simple synthetic "real" data generator.

Usage example:
    python train_mara_panna.py --epochs 200 --batch-size 32 --device cuda --outdir ./checkpoints

Requirements:
    pip install torch torchvision numpy tqdm

Notes:
 - This is intentionally self-contained and uses a synthetic real-data generator so it runs without external datasets.
 - For production, swap `RealThreatDataset` with your real dataset loader.
"""

from __future__ import annotations
import os
import json
import math
import time
import argparse
from typing import Tuple, Dict, Any, List
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Models: MaraGenerator and PannaDiscriminator
# (Architectures aligned with your node-agent definitions but training-friendly)
# -------------------------------
class MaraGenerator(nn.Module):
    def __init__(self, latent_dim: int = 128, seq_len: int = 24, features: int = 12):
        super().__init__()
        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, seq_len * features),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return shape [batch, seq_len, features] with values in [-1,1]"""
        out = self.net(z)
        return out.view(-1, self.seq_len, self.features)

class PannaDiscriminator(nn.Module):
    def __init__(self, seq_len: int = 24, features: int = 12):
        super().__init__()
        in_dim = seq_len * features
        self.seq_len = seq_len
        self.features = features
        # Output a single scalar (probability of real)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        return self.net(flat).view(-1)

# -------------------------------
# Synthetic "real" data generator (for demo / Baydin test)
# Replace with real OSINT-derived sequences for production.
# -------------------------------
class RealThreatDataset(Dataset):
    """
    Produces synthetic 'real' threat sequences that mimic structured patterns:
    mixture of sinusoids + localized spikes + correlated noise across features.
    """
    def __init__(self, size: int = 5000, seq_len: int = 24, features: int = 12, seed: int = 42):
        super().__init__()
        self.size = size
        self.seq_len = seq_len
        self.features = features
        self.rng = np.random.RandomState(seed)
        self.data = [self._sample() for _ in range(size)]

    def _sample(self) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, self.seq_len)
        base = np.stack([
            np.sin(t * (1 + self.rng.uniform(-0.3, 0.3))) for _ in range(self.features)
        ], axis=-1)  # shape [seq_len, features]
        # add localized spikes
        for _ in range(self.rng.randint(1, 4)):
            pos = self.rng.randint(0, self.seq_len)
            amp = self.rng.uniform(0.5, 1.5)
            base[pos:pos + 1, :] += amp * (self.rng.normal(size=(1, self.features)))
        # correlated noise
        noise = 0.15 * self.rng.normal(size=(self.seq_len, self.features))
        sample = base + noise
        # normalize to [-1,1]
        sample = sample / (np.max(np.abs(sample)) + 1e-9)
        return sample.astype(np.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

# -------------------------------
# Utilities: checkpoints, metrics, seeding
# -------------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state: Dict[str, Any], outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    torch.save(state, str(path))
    print(f"[CHECKPOINT] saved: {path}")

def load_checkpoint(path: Path, device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
    checkpoint = torch.load(str(path), map_location=device)
    print(f"[CHECKPOINT] loaded: {path}")
    return checkpoint

# -------------------------------
# Baydin self-test helpers
# -------------------------------
def compute_diversity_metric(samples: torch.Tensor) -> float:
    """
    Diversity metric: mean std across features and time.
    Input shape: [n_samples, seq_len, features]
    """
    with torch.no_grad():
        # flatten per-sample
        s = samples.view(samples.size(0), -1)
        stds = torch.std(s, dim=0)  # per-dim std
        return float(torch.mean(stds).item())

def baydin_self_test(generator: nn.Module,
                     discriminator: nn.Module,
                     device: torch.device,
                     n_samples: int = 128,
                     latent_dim: int = 128,
                     val_loader: DataLoader = None) -> Dict[str, Any]:
    """
    Runs a short, automated adversarial stress validation:
      - generates samples and measures diversity,
      - evaluates discriminator accuracy on real val set (if provided),
      - computes generator-based fooling ratio.
    Returns a dictionary with metrics to judge health.
    """
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        gen = generator(z)  # [n, seq_len, features]
        diversity = compute_diversity_metric(gen)

        # Discriminator 'fooling ratio': fraction of generated samples classified as real (>0.5)
        preds_gen = discriminator(gen.to(device))
        fool_ratio = float((preds_gen > 0.5).float().mean().item())

        val_acc = None
        if val_loader is not None:
            total = 0
            correct = 0
            for batch in val_loader:
                real = batch.to(device)
                preds = discriminator(real)
                # real labeled 1
                correct += (preds > 0.5).sum().item()
                total += real.size(0)
            val_acc = float(correct / total) if total > 0 else None

    generator.train()
    discriminator.train()

    return {"diversity": diversity, "fool_ratio": fool_ratio, "val_acc": val_acc}

# -------------------------------
# Training harness
# -------------------------------
def train(args: argparse.Namespace):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"[TRAIN] device: {device}")

    set_seed(args.seed)

    # Prepare models
    G = MaraGenerator(latent_dim=args.latent_dim, seq_len=args.seq_len, features=args.features).to(device)
    D = PannaDiscriminator(seq_len=args.seq_len, features=args.features).to(device)

    if args.load_checkpoint:
        ck = load_checkpoint(Path(args.load_checkpoint), device=device)
        G.load_state_dict(ck["G_state"])
        D.load_state_dict(ck["D_state"])
        start_epoch = ck.get("epoch", 0) + 1
        print(f"[TRAIN] resumed from epoch {start_epoch}")
    else:
        start_epoch = 1

    # Optimizers
    opt_G = optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999))

    # Optional schedulers
    sched_G = optim.lr_scheduler.StepLR(opt_G, step_size=args.lr_step, gamma=0.5) if args.lr_step > 0 else None
    sched_D = optim.lr_scheduler.StepLR(opt_D, step_size=args.lr_step, gamma=0.5) if args.lr_step > 0 else None

    # Loss
    criterion = nn.BCELoss()

    # Data
    dataset = RealThreatDataset(size=args.train_size, seq_len=args.seq_len, features=args.features, seed=args.seed)
    valset = RealThreatDataset(size=args.val_size, seq_len=args.seq_len, features=args.features, seed=args.seed + 1)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # tracking
    history: List[Dict[str, Any]] = []
    best_metric = float("inf")
    no_improve_epochs = 0

    print("[TRAIN] starting loop")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        epoch_metrics = {"epoch": epoch, "time": None, "D_loss": 0.0, "G_loss": 0.0, "D_acc": 0.0}
        running_D_loss = 0.0
        running_G_loss = 0.0
        running_D_correct = 0
        running_D_total = 0

        G.train()
        D.train()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for real_batch in pbar:
            real = real_batch.to(device)  # [B, seq_len, features]
            bsize = real.size(0)

            # ----------------------------
            # Train Discriminator: real=1, fake=0
            # ----------------------------
            opt_D.zero_grad()
            labels_real = torch.ones(bsize, device=device)
            labels_fake = torch.zeros(bsize, device=device)

            # real loss
            preds_real = D(real)
            loss_real = criterion(preds_real, labels_real)

            # fake loss
            z = torch.randn(bsize, args.latent_dim, device=device)
            fake = G(z).detach()  # detach so G not updated here
            preds_fake = D(fake)
            loss_fake = criterion(preds_fake, labels_fake)

            loss_D = (loss_real + loss_fake) * 0.5
            # optional gradient penalty (simple L2 on weights) to stabilise
            if args.grad_penalty > 0:
                gp = 0.0
                for p in D.parameters():
                    gp += (p.pow(2).sum()) * 1e-6
                loss_D = loss_D + gp

            loss_D.backward()
            opt_D.step()

            # tracking discriminator accuracy on this mini-batch
            preds_all = torch.cat([preds_real, preds_fake], dim=0)
            labels_all = torch.cat([labels_real, labels_fake], dim=0)
            predicted = (preds_all > 0.5).float()
            running_D_correct += (predicted == labels_all).sum().item()
            running_D_total += labels_all.numel()

            # ----------------------------
            # Train Generator: want D(fake) -> 1
            # ----------------------------
            opt_G.zero_grad()
            z2 = torch.randn(bsize, args.latent_dim, device=device)
            fake2 = G(z2)
            preds_for_G = D(fake2)
            loss_G = criterion(preds_for_G, labels_real)  # trick: want discriminator to label fake as real

            # optional feature-match or similarity regularizer
            if args.fm_weight > 0:
                # simple L1 between feature means of fake and real
                fm = torch.mean(torch.abs(fake2.mean(dim=0) - real.mean(dim=0)))
                loss_G = loss_G + args.fm_weight * fm

            loss_G.backward()
            opt_G.step()

            running_D_loss += loss_D.item()
            running_G_loss += loss_G.item()

            pbar.set_postfix({"D_loss": running_D_loss / (1 + pbar.n), "G_loss": running_G_loss / (1 + pbar.n)})

        # epoch stats
        epoch_metrics["D_loss"] = running_D_loss / len(dataloader)
        epoch_metrics["G_loss"] = running_G_loss / len(dataloader)
        epoch_metrics["D_acc"] = running_D_correct / running_D_total if running_D_total > 0 else None
        epoch_metrics["time"] = time.time() - t0

        # scheduler step
        if sched_G:
            sched_G.step()
        if sched_D:
            sched_D.step()

        # Baydin self-test (periodic)
        if epoch % args.baydin_interval == 0:
            baydin_metrics = baydin_self_test(G, D, device, n_samples=args.baydin_samples,
                                             latent_dim=args.latent_dim, val_loader=valloader)
            epoch_metrics.update({"baydin": baydin_metrics})
            # evaluate for early stopping: use val_acc if available, else D_loss
            monitor = baydin_metrics.get("val_acc")
            if monitor is None:
                monitor = epoch_metrics["D_loss"]
            # early stop heuristics
            # - if diversity very low and fool_ratio near 1.0 -> mode collapse
            if baydin_metrics["diversity"] < args.baydin_diversity_thresh and baydin_metrics["fool_ratio"] > args.baydin_fool_thresh:
                print(f"[BAYDIN WARNING] potential mode collapse detected (div={baydin_metrics['diversity']:.6f}, fool={baydin_metrics['fool_ratio']:.3f})")
                # optionally reduce LR
                for g in opt_G.param_groups:
                    g["lr"] *= 0.5
                for g in opt_D.param_groups:
                    g["lr"] *= 0.5

        # save metrics
        history.append(epoch_metrics)
        # print summary
        print(f"Epoch {epoch} | D_loss {epoch_metrics['D_loss']:.4f} | G_loss {epoch_metrics['G_loss']:.4f} | D_acc {epoch_metrics['D_acc']:.3f} | time {epoch_metrics['time']:.1f}s")

        # checkpointing: save latest and best by D_loss
        ckpt = {
            "epoch": epoch,
            "G_state": G.state_dict(),
            "D_state": D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "sched_G": sched_G.state_dict
