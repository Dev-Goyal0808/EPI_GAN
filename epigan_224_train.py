#!/usr/bin/env python3
"""
Epi-GAN 224x224 retraining pipeline for HAM10000.

What this script does
---------------------
1. Reads GroundTruth.csv and matches image_id to images/<image_id>.jpg.
2. Splits real images BEFORE GAN training (train/val/test); if a lesion_id column
   exists, it uses group-level splitting to reduce lesion leakage. GroundTruth.csv
   often does not contain lesion_id; in that case image-level stratification is used
   and the script prints a warning. For a publication-grade lesion-level split, merge
   HAM10000_metadata.csv and split by lesion_id.
3. Copies real images into dataset_split/{train,val,test}/<CLASS>/ using pathlib.
4. Trains ONE class-conditional GAN using TRAIN images only, at 224x224.
5. Includes FRB (Feature Refinement Block) and FPB (Feature Penetration Block)
   explicitly in the generator. These definitions are operational choices and must
   be described consistently in the manuscript.
6. Applies lightweight differentiable augmentation to real/fake discriminator inputs.
7. Evaluates FID and KID against held-out REAL validation images at a fixed resolution,
   logs the sample counts, and saves continuous FID/KID plots.
8. Generates additional synthetic images ONLY inside train/<CLASS>/augmented_epigan/
   to balance each training class to the largest training class. Validation/test
   folders are never augmented.

IMPORTANT
---------
- Do not use test images to train the GAN, select checkpoints, or calculate FID/KID.
- This script uses validation images as the reference set for generative metrics.
  The test set remains real-only for downstream classifier evaluation.
- FID/KID are computed from ImageNet InceptionV3 pool features at 299x299, even though
  the GAN output is 224x224. This is standard for this implementation; report it.
- KID is an unbiased polynomial-kernel MMD estimator. FID is estimated from the
  available feature samples and can be noisy for small classes; interpret class-wise
  values cautiously.
- If pretrained Inception weights are unavailable and cannot be downloaded, metric
  evaluation will fail; ensure the server can access torchvision weights or pre-cache
  them.
- A 224x224 GAN is more expensive than the prior 64x64 model. Start with batch size 32
  on one GPU; this server has ample VRAM, but actual available memory may be constrained
  by MIG partitions. Increase batch size only after confirming `nvidia-smi`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_path(image_root: Path, image_id: str) -> Optional[Path]:
    image_id = str(image_id).strip()
    candidates = [
        image_root / f"{image_id}.jpg",
        image_root / f"{image_id}.jpeg",
        image_root / f"{image_id}.png",
        image_root / image_id,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_labels(csv_path: Path, image_root: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)
    if "image_id" not in df.columns:
        # Common alternative column name
        alternatives = [c for c in df.columns if c.lower() in {"image", "image_id", "isic_id"}]
        if not alternatives:
            raise ValueError(
                f"Could not find an image_id column in {csv_path}. "
                f"Columns found: {list(df.columns)}"
            )
        df = df.rename(columns={alternatives[0]: "image_id"})

    missing_classes = [c for c in CLASSES if c not in df.columns]
    if missing_classes:
        raise ValueError(
            f"GroundTruth CSV is missing class columns {missing_classes}. "
            f"Expected one-hot columns: {CLASSES}"
        )

    # GroundTruth is expected to be one-hot encoded. Drop rows with ambiguous labels.
    class_matrix = df[CLASSES].apply(pd.to_numeric, errors="coerce").fillna(0)
    valid = class_matrix.sum(axis=1).eq(1)
    df = df.loc[valid].copy()
    class_matrix = class_matrix.loc[valid]
    df["label"] = class_matrix.idxmax(axis=1)

    paths, keep = [], []
    for idx, row in df.iterrows():
        p = resolve_image_path(image_root, row["image_id"])
        if p is not None:
            paths.append(str(p.resolve()))
            keep.append(idx)
    df = df.loc[keep].copy()
    df["image_path"] = paths
    df = df.reset_index(drop=True)
    if df.empty:
        raise RuntimeError(
            f"No images matched CSV IDs under {image_root}. "
            "Check --images-dir and the image file extensions."
        )
    return df, sorted(df["label"].unique().tolist())


def split_dataframe(df: pd.DataFrame, seed: int):
    """Split real images before any GAN training; use lesion groups if available."""
    # GroundTruth.csv usually has no lesion_id. If present, avoid group leakage.
    group_col = next((c for c in ["lesion_id", "lesionId", "lesion"] if c in df.columns), None)
    if group_col:
        try:
            from sklearn.model_selection import GroupShuffleSplit
            # Approximate 80/10/10 group split. GroupShuffleSplit does not stratify.
            gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
            train_idx, temp_idx = next(gss1.split(df, groups=df[group_col]))
            train_df, temp_df = df.iloc[train_idx].copy(), df.iloc[temp_idx].copy()
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
            val_rel, test_rel = next(gss2.split(temp_df, groups=temp_df[group_col]))
            val_df, test_df = temp_df.iloc[val_rel].copy(), temp_df.iloc[test_rel].copy()
            print(f"[SPLIT] Group/lesion-level split using column '{group_col}'.")
            return train_df, val_df, test_df, "lesion/group-level"
        except Exception as exc:
            print(f"[WARN] Group split failed ({exc}); falling back to stratified image split.")

    print(
        "[WARN] No usable lesion_id column was found in GroundTruth.csv. "
        "Using image-level stratified splitting. For the paper, merge HAM10000_metadata.csv "
        "and rerun with lesion_id available to prevent same-lesion leakage."
    )
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"]
    )
    return train_df.copy(), val_df.copy(), test_df.copy(), "image-level stratified"


def materialize_splits(train_df, val_df, test_df, output_root: Path, force: bool):
    split_root = output_root / "dataset_split"
    if split_root.exists() and force:
        shutil.rmtree(split_root)
    for split_name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for label in CLASSES:
            (split_root / split_name / label).mkdir(parents=True, exist_ok=True)
        for _, row in frame.iterrows():
            src = Path(row["image_path"])
            dst = split_root / split_name / row["label"] / f"{row['image_id']}{src.suffix.lower()}"
            if not dst.exists():
                shutil.copy2(src, dst)
    return split_root


def save_split_counts(train_df, val_df, test_df, output_root: Path, split_kind: str, seed: int):
    rows = []
    for split_name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = frame["label"].value_counts()
        for cls in CLASSES:
            rows.append({"split": split_name, "class": cls, "real_images": int(counts.get(cls, 0))})
    pd.DataFrame(rows).to_csv(output_root / "split_class_counts.csv", index=False)
    (output_root / "split_protocol.json").write_text(
        json.dumps({
            "split_type": split_kind,
            "seed": int(seed),
            "train_fraction_approx": 0.8,
            "validation_fraction_approx": 0.1,
            "test_fraction_approx": 0.1,
            "gan_training_images": "train split only",
            "test_images": "real images only; never used for GAN training or metric selection",
        }, indent=2),
        encoding="utf-8",
    )


class SkinDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, class_to_idx: Dict[str, int], size: int = 224):
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as im:
            image = self.transform(im.convert("RGB"))
        return image, self.class_to_idx[row["label"]]


class FRB(nn.Module):
    """Feature Refinement Block: residual local texture/detail refinement."""
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return F.leaky_relu(x + self.gate * self.body(x), 0.2, inplace=True)


class FPB(nn.Module):
    """Feature Penetration Block: multi-receptive-field feature fusion."""
    def __init__(self, channels: int):
        super().__init__()
        branch = max(channels // 4, 16)
        self.b1 = nn.Conv2d(channels, branch, 3, padding=1, dilation=1)
        self.b2 = nn.Conv2d(channels, branch, 3, padding=2, dilation=2)
        self.b3 = nn.Conv2d(channels, branch, 3, padding=3, dilation=3)
        self.fuse = nn.Sequential(
            nn.Conv2d(branch * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        multi = torch.cat([
            F.leaky_relu(self.b1(x), 0.2, inplace=True),
            F.leaky_relu(self.b2(x), 0.2, inplace=True),
            F.leaky_relu(self.b3(x), 0.2, inplace=True),
        ], dim=1)
        return F.leaky_relu(x + self.gate * self.fuse(multi), 0.2, inplace=True)


class ConditionalGenerator(nn.Module):
    """Class-conditional generator: 7x7 -> 14 -> 28 -> 56 -> 112 -> 224."""
    def __init__(self, latent_dim=128, num_classes=7, embed_dim=128, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Linear(latent_dim + embed_dim, base * 8 * 7 * 7)
        self.project = nn.Sequential(
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            FRB(base * 8),
            FPB(base * 8),
        )
        channels = [base * 8, base * 4, base * 2, base, max(base // 2, 16), max(base // 4, 8)]
        blocks = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            blocks += [
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            # Apply FRB/FPB at 28x28 and 112x112 feature scales, plus final refinement.
            if out_c in (base * 2, max(base // 2, 16)):
                blocks += [FRB(out_c), FPB(out_c)]
        self.upsample = nn.Sequential(*blocks)
        final_c = channels[-1]
        self.to_rgb = nn.Sequential(
            nn.Conv2d(final_c, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        emb = self.label_emb(labels)
        x = self.fc(torch.cat([z, emb], dim=1)).view(z.shape[0], -1, 7, 7)
        x = self.project(x)
        x = self.upsample(x)
        return self.to_rgb(x)


class ConditionalDiscriminator(nn.Module):
    """Projection discriminator with class embedding; input is 224x224x3."""
    def __init__(self, num_classes=7, base=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.1), #112
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.1), #56
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.1), #28
            nn.Conv2d(base * 4, base * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True), #14
            nn.Conv2d(base * 8, base * 8, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), #7
            nn.AdaptiveAvgPool2d(1),
        )
        self.unconditional = nn.Linear(base * 8, 1)
        self.class_embedding = nn.Embedding(num_classes, base * 8)

    def forward(self, image, labels):
        feat = self.features(image).flatten(1)
        projection = (feat * self.class_embedding(labels)).sum(dim=1, keepdim=True) / math.sqrt(feat.shape[1])
        return self.unconditional(feat) + projection


def diff_augment(x):
    """Small differentiable augmentations applied to D inputs, not to saved images."""
    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])
    if random.random() < 0.5:
        shift_y = random.randint(-8, 8)
        shift_x = random.randint(-8, 8)
        x = torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))
    if random.random() < 0.5:
        gain = 0.9 + 0.2 * torch.rand(x.shape[0], 1, 1, 1, device=x.device)
        x = x * gain
    return x.clamp(-1, 1)


class InceptionFeatures(nn.Module):
    def __init__(self, device):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, transform_input=False, aux_logits=True)
        model.fc = nn.Identity()
        model.eval()
        self.model = model.to(device)
        self.device = device
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        # GAN tensors [-1,1]; Inception expects normalized RGB.
        x = (x + 1) / 2
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, dict):
            out = out["pool"]
        return out.flatten(1).float()


def collect_real_metric_images(frame, class_to_idx, per_class_limit, device):
    """Load held-out validation images, capped equally per class for comparability."""
    rows = []
    for cls in CLASSES:
        subset = frame[frame["label"] == cls]
        if len(subset) > per_class_limit:
            subset = subset.sample(n=per_class_limit, random_state=123)
        rows.extend([(row["image_path"], class_to_idx[cls]) for _, row in subset.iterrows()])
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    images, labels = [], []
    for path, label in rows:
        try:
            with Image.open(path) as im:
                images.append(transform(im.convert("RGB")))
                labels.append(label)
        except Exception:
            continue
    if not images:
        raise RuntimeError("No validation images available for FID/KID.")
    return torch.stack(images).to(device), torch.tensor(labels, device=device)


def fid_from_features(real: np.ndarray, fake: np.ndarray) -> float:
    real_mu, fake_mu = real.mean(0), fake.mean(0)
    real_cov = np.cov(real, rowvar=False)
    fake_cov = np.cov(fake, rowvar=False)
    covmean = sqrtm(real_cov @ fake_cov)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = float(np.sum((real_mu - fake_mu) ** 2) + np.trace(real_cov + fake_cov - 2 * covmean))
    return max(value, 0.0)


def kid_from_features(real: np.ndarray, fake: np.ndarray, subsets=20, subset_size=1000) -> Tuple[float, float]:
    """Unbiased KID estimate (polynomial kernel), returned as mean and SD across subsets."""
    n = min(len(real), len(fake), subset_size)
    if n < 2:
        return float("nan"), float("nan")
    d = real.shape[1]
    values = []
    rng = np.random.default_rng(12345)
    for _ in range(subsets):
        rx = real[rng.choice(len(real), n, replace=False)]
        fx = fake[rng.choice(len(fake), n, replace=False)]
        k_xx = (rx @ rx.T / d + 1) ** 3
        k_yy = (fx @ fx.T / d + 1) ** 3
        k_xy = (rx @ fx.T / d + 1) ** 3
        np.fill_diagonal(k_xx, 0)
        np.fill_diagonal(k_yy, 0)
        mmd = k_xx.sum() / (n * (n - 1)) + k_yy.sum() / (n * (n - 1)) - 2 * k_xy.mean()
        values.append(float(mmd))
    return float(np.mean(values)), float(np.std(values, ddof=1) if len(values) > 1 else 0.0)


@torch.no_grad()
def evaluate_fid_kid(generator, feature_net, real_images, real_labels, latent_dim, num_classes, batch_size, device):
    generator.eval()
    # Generate the same number of samples as validation reference, label-balanced.
    n = len(real_images)
    labels = torch.arange(n, device=device) % num_classes
    # Shuffle the class labels so metric set is not ordered by class.
    labels = labels[torch.randperm(n, device=device)]
    fake_parts = []
    for start in range(0, n, batch_size):
        y = labels[start:start + batch_size]
        z = torch.randn(len(y), latent_dim, device=device)
        fake_parts.append(generator(z, y).clamp(-1, 1))
    fake_images = torch.cat(fake_parts, dim=0)
    real_feats, fake_feats = [], []
    for start in range(0, n, batch_size):
        real_feats.append(feature_net(real_images[start:start + batch_size]).cpu().numpy())
        fake_feats.append(feature_net(fake_images[start:start + batch_size]).cpu().numpy())
    real_feats = np.concatenate(real_feats, axis=0)
    fake_feats = np.concatenate(fake_feats, axis=0)
    fid = fid_from_features(real_feats, fake_feats)
    kid_mean, kid_std = kid_from_features(real_feats, fake_feats)
    generator.train()
    return fid, kid_mean, kid_std, n


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU VRAM] {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    csv_path = Path(args.groundtruth).expanduser().resolve()
    image_root = Path(args.images_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    df, _ = read_labels(csv_path, image_root)
    train_df, val_df, test_df, split_kind = split_dataframe(df, args.seed)
    # Set a split-specific seed in metadata rather than hard-code the value.
    split_root = materialize_splits(train_df, val_df, test_df, output_root, args.force_rebuild_splits)
    save_split_counts(train_df, val_df, test_df, output_root, split_kind, args.seed)
    print(f"[SPLIT] {split_kind}; real images: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print("[TRAIN COUNTS]", train_df["label"].value_counts().reindex(CLASSES, fill_value=0).to_dict())
    print("[VAL COUNTS]", val_df["label"].value_counts().reindex(CLASSES, fill_value=0).to_dict())
    print("[TEST COUNTS]", test_df["label"].value_counts().reindex(CLASSES, fill_value=0).to_dict())

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    train_ds = SkinDataset(train_df, class_to_idx, 224)
    # Balanced class sampling prevents NV frequency from dominating conditional GAN updates.
    train_labels = train_df["label"].map(class_to_idx).to_numpy()
    class_counts = np.bincount(train_labels, minlength=len(CLASSES)).astype(np.float64)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_ds), replacement=True
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError("Training DataLoader has zero batches. Reduce --batch-size.")

    latent_dim = args.latent_dim
    G = ConditionalGenerator(latent_dim, len(CLASSES), base=args.g_base).to(device)
    D = ConditionalDiscriminator(len(CLASSES), base=args.d_base).to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.99))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.99))
    scaler_g = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    scaler_d = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    use_amp = args.amp and device.type == "cuda"

    # Validation real images only, never the test set.
    metric_real, metric_labels = collect_real_metric_images(
        val_df, class_to_idx, args.metric_samples_per_class, device
    )
    metric_net = InceptionFeatures(device)
    history = []
    best_fid = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        g_loss_sum = d_loss_sum = 0.0
        n_batches = 0
        for real, labels in loader:
            real = real.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bs = real.shape[0]
            # Train discriminator with matching real/fake class labels.
            z = torch.randn(bs, latent_dim, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                fake = G(z, labels).detach()
                real_aug = diff_augment(real)
                fake_aug = diff_augment(fake)
                real_score = D(real_aug, labels)
                fake_score = D(fake_aug, labels)
                # Hinge loss is generally more stable than BCE for GAN training.
                d_loss = F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
            d_opt.zero_grad(set_to_none=True)
            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(d_opt)
            torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
            scaler_d.step(d_opt)
            scaler_d.update()

            z = torch.randn(bs, latent_dim, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                fake = G(z, labels)
                g_loss = -D(diff_augment(fake), labels).mean()
            g_opt.zero_grad(set_to_none=True)
            scaler_g.scale(g_loss).backward()
            scaler_g.unscale_(g_opt)
            torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
            scaler_g.step(g_opt)
            scaler_g.update()

            g_loss_sum += float(g_loss.detach())
            d_loss_sum += float(d_loss.detach())
            n_batches += 1

        row = {
            "epoch": epoch,
            "g_loss": g_loss_sum / max(n_batches, 1),
            "d_loss": d_loss_sum / max(n_batches, 1),
            "fid": float("nan"),
            "kid_mean": float("nan"),
            "kid_std": float("nan"),
            "metric_n_per_distribution": len(metric_real),
        }
        print(
            f"Epoch {epoch:04d}/{args.epochs} | "
            f"G={row['g_loss']:.4f} D={row['d_loss']:.4f}",
            end="",
        )
        if epoch == 1 or epoch % args.metrics_every == 0 or epoch == args.epochs:
            try:
                fid, kid_mean, kid_std, metric_n = evaluate_fid_kid(
                    G, metric_net, metric_real, metric_labels, latent_dim,
                    len(CLASSES), args.metric_batch_size, device
                )
                row.update(fid=fid, kid_mean=kid_mean, kid_std=kid_std, metric_n_per_distribution=metric_n)
                print(f" | FID={fid:.3f} | KID={kid_mean:.6f} ± {kid_std:.6f} (n={metric_n})")
                if fid < best_fid:
                    best_fid = fid
                    torch.save({
                        "epoch": epoch,
                        "generator": G.state_dict(),
                        "discriminator": D.state_dict(),
                        "class_to_idx": class_to_idx,
                        "config": vars(args),
                        "fid": fid,
                        "kid_mean": kid_mean,
                        "kid_std": kid_std,
                    }, output_root / "epigan_224_best_fid.pt")
            except Exception as exc:
                print(f" | METRIC ERROR: {exc}")
                print("  Check torchvision/Inception weights and validation sample count.")
        else:
            print()

        history.append(row)
        pd.DataFrame(history).to_csv(output_root / "training_metrics.csv", index=False)
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save({
                "epoch": epoch,
                "generator": G.state_dict(),
                "discriminator": D.state_dict(),
                "class_to_idx": class_to_idx,
                "config": vars(args),
            }, output_root / f"epigan_224_epoch_{epoch:04d}.pt")
        if epoch % args.sample_every == 0 or epoch == 1:
            save_sample_grid(G, output_root / "samples", epoch, latent_dim, len(CLASSES), device)

    # Plot continuous metric histories. Missing evaluations remain NaN.
    plot_training_curves(pd.DataFrame(history), output_root)
    save_augmented_training_images(
        G, train_df, class_to_idx, output_root, split_root, latent_dim, device, args
    )
    print(f"[DONE] Elapsed: {(time.time() - start_time) / 3600:.2f} h")
    print(f"[DONE] Outputs: {output_root}")
    print("[IMPORTANT] Downstream classifier test data must be split_root/test only (real images).")


@torch.no_grad()
def save_sample_grid(generator, output_dir: Path, epoch: int, latent_dim: int, num_classes: int, device):
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    labels = torch.arange(num_classes, device=device).repeat_interleave(2)
    z = torch.randn(len(labels), latent_dim, device=device)
    images = ((generator(z, labels).clamp(-1, 1) + 1) / 2).cpu()
    fig, axes = plt.subplots(num_classes, 2, figsize=(5, 2.5 * num_classes))
    for cls_idx in range(num_classes):
        for j in range(2):
            ax = axes[cls_idx, j]
            im = images[cls_idx * 2 + j].permute(1, 2, 0).numpy()
            ax.imshow(im)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(CLASSES[cls_idx])
    fig.suptitle(f"Epi-GAN 224x224 samples — epoch {epoch}")
    fig.tight_layout()
    fig.savefig(output_dir / f"samples_epoch_{epoch:04d}.png", dpi=150)
    plt.close(fig)
    generator.train()


def plot_training_curves(history: pd.DataFrame, output_root: Path):
    if history.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["epoch"], history["g_loss"], label="Generator loss")
    ax.plot(history["epoch"], history["d_loss"], label="Discriminator loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Epi-GAN 224x224 training losses")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "loss_curves.png", dpi=160)
    plt.close(fig)

    metric_rows = history.dropna(subset=["fid", "kid_mean"])
    if len(metric_rows):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(metric_rows["epoch"], metric_rows["fid"], marker="o", label="FID")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("FID (lower is better)")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(metric_rows["epoch"], metric_rows["kid_mean"], marker="s", label="KID")
        ax2.set_ylabel("KID (lower is better; can be negative as an unbiased estimate)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        ax1.set_title("Held-out validation FID and KID over training")
        fig.tight_layout()
        fig.savefig(output_root / "fid_kid_continuous.png", dpi=180)
        plt.close(fig)


@torch.no_grad()
def save_augmented_training_images(generator, train_df, class_to_idx, output_root, split_root,
                                   latent_dim, device, args):
    generator.eval()
    train_counts = train_df["label"].value_counts().reindex(CLASSES, fill_value=0).to_dict()
    target = max(train_counts.values())
    aug_root = split_root / "train"
    rows = []
    for cls in CLASSES:
        current = int(train_counts[cls])
        needed = max(0, target - current) if args.balance_train else args.images_per_class
        if not args.balance_train:
            needed = args.images_per_class
        class_dir = aug_root / cls / "augmented_epigan"
        class_dir.mkdir(parents=True, exist_ok=True)
        # Remove only previous outputs created by this script.
        for p in class_dir.glob("epigan_*.png"):
            p.unlink()
        saved = 0
        for start in range(0, needed, args.generate_batch_size):
            count = min(args.generate_batch_size, needed - start)
            labels = torch.full((count,), class_to_idx[cls], device=device, dtype=torch.long)
            z = torch.randn(count, latent_dim, device=device)
            generated = ((generator(z, labels).clamp(-1, 1) + 1) * 127.5).round().byte().cpu()
            for j in range(count):
                arr = generated[j].permute(1, 2, 0).numpy()
                Image.fromarray(arr, mode="RGB").save(class_dir / f"epigan_{start+j:06d}.png")
                saved += 1
        rows.append({
            "class": cls,
            "real_train_images": current,
            "synthetic_train_images": saved,
            "post_augmentation_train_total": current + saved,
            "real_to_synthetic_ratio": (f"{current}:{saved}" if saved else f"{current}:0"),
        })
        print(f"[GENERATE] {cls}: real train={current}, synthetic={saved}, post={current+saved}")
    pd.DataFrame(rows).to_csv(output_root / "post_augmentation_class_counts.csv", index=False)
    generator.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Train class-conditional Epi-GAN at 224x224.")
    parser.add_argument("--groundtruth", type=str, default="dataset/GroundTruth.csv")
    parser.add_argument("--images-dir", type=str, default="dataset/images")
    parser.add_argument("--output-dir", type=str, default="runs/epigan_224")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--g-base", type=int, default=64)
    parser.add_argument("--d-base", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--metrics-every", type=int, default=5)
    parser.add_argument("--metric-samples-per-class", type=int, default=256)
    parser.add_argument("--metric-batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--generate-batch-size", type=int, default=64)
    parser.add_argument("--balance-train", action="store_true",
                        help="Generate enough minority-class samples to match the largest training class.")
    parser.add_argument("--images-per-class", type=int, default=1000,
                        help="Used if --balance-train is not specified.")
    parser.add_argument("--force-rebuild-splits", action="store_true",
                        help="Delete and recreate output/dataset_split.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
