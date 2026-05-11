"""
GAN from Scratch — DCGAN on FashionMNIST
적대적 학습(Adversarial Training)을 통해 28×28 이미지 생성을 학습합니다.

References:
    - Goodfellow et al., "Generative Adversarial Networks" (NeurIPS 2014)
        https://arxiv.org/abs/1406.2661
    - Radford et al., "Unsupervised Representation Learning with Deep
        Convolutional Generative Adversarial Networks" (ICLR 2016)
        https://arxiv.org/abs/1511.06434
"""

import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# matplotlib 한글 폰트 적용
_korean_fonts = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic", "AppleGothic", "Noto Sans CJK KR", "Gulim"]
_available = {f.name for f in fm.fontManager.ttflist}
for _font in _korean_fonts:
    if _font in _available:
        mpl.rcParams["font.family"] = _font
        break
mpl.rcParams["axes.unicode_minus"] = False


# ───────────────────────────────────────────────────────────
# 1. 경로 / 시드 / 디바이스
# ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)


# ───────────────────────────────────────────────────────────
# 2. 하이퍼파라미터 (DCGAN 표준)
# ───────────────────────────────────────────────────────────
Z_DIM = 100                    # latent vector dimension
G_CHANNELS = (128, 64)         # Generator hidden channels (deeper → shallower)
D_CHANNELS = (64, 128)         # Discriminator hidden channels (shallower → deeper)
BATCH_SIZE = 128
EPOCHS = 50
LR = 2e-4                      # Adam learning rate (DCGAN paper)
BETA1 = 0.5                    # Adam β1 (DCGAN paper, 안정적인 학습)

FIXED_Z_SAMPLES = 8            # epoch별 학습 진행 모니터링용 z 개수
SNAPSHOT_EPOCHS = [1, 10, 25, 50]  # progression 시각화에 쓸 epoch

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ───────────────────────────────────────────────────────────
# 3. 데이터 (FashionMNIST, normalized to [-1, 1])
# ───────────────────────────────────────────────────────────
def get_dataloader(batch_size=BATCH_SIZE):
    """FashionMNIST를 [-1, 1] 범위로 정규화 (Generator의 Tanh output 매칭)."""
    transform = transforms.Compose([
        transforms.ToTensor(),                 # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]
    ])
    train_set = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    return train_loader, train_set


# ───────────────────────────────────────────────────────────
# 4. Generator (DCGAN)
# ───────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    DCGAN Generator: z(100) → 28×28 image

    구조:
        FC(100 → 7×7×128) → BatchNorm → ReLU
        Reshape (128, 7, 7)
        ConvT(128→64, stride=2) → BatchNorm → ReLU    # 7×7 → 14×14
        ConvT(64→1,  stride=2) → Tanh                 # 14×14 → 28×28
    """
    def __init__(self, z_dim=Z_DIM, hidden_channels=G_CHANNELS):
        super().__init__()
        c1, c2 = hidden_channels
        self.c1 = c1
        self.fc = nn.Linear(z_dim, c1 * 7 * 7)
        self.bn_fc = nn.BatchNorm1d(c1 * 7 * 7)
        self.convt1 = nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(c2)
        self.convt2 = nn.ConvTranspose2d(c2, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        h = F.relu(self.bn_fc(self.fc(z)))
        h = h.view(-1, self.c1, 7, 7)
        h = F.relu(self.bn1(self.convt1(h)))
        return torch.tanh(self.convt2(h))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ───────────────────────────────────────────────────────────
# 5. Discriminator (DCGAN)
# ───────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    DCGAN Discriminator: 28×28 image → real(1) / fake(0) logit

    구조:
        Conv(1→64,  stride=2) → LeakyReLU(0.2)             # 28×28 → 14×14
        Conv(64→128, stride=2) → BatchNorm → LeakyReLU(0.2) # 14×14 → 7×7
        Flatten → FC(128×7×7 → 1)
        (Sigmoid는 BCEWithLogitsLoss에서 적용)
    """
    def __init__(self, hidden_channels=D_CHANNELS):
        super().__init__()
        c1, c2 = hidden_channels
        self.conv1 = nn.Conv2d(1, c1, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.fc = nn.Linear(c2 * 7 * 7, 1)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.2)
        h = h.flatten(1)
        return self.fc(h)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ───────────────────────────────────────────────────────────
# 6. VAE inline 정의 (06번 시각화 비교용 — #29 호환)
# ───────────────────────────────────────────────────────────
VAE_LATENT_DIM = 16
VAE_HIDDEN_CHANNELS = (32, 64)


class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=VAE_LATENT_DIM, hidden_channels=VAE_HIDDEN_CHANNELS):
        super().__init__()
        c1, c2 = hidden_channels
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.flatten_dim = c2 * 7 * 7
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=VAE_LATENT_DIM, hidden_channels=VAE_HIDDEN_CHANNELS):
        super().__init__()
        c1, c2 = hidden_channels
        self.fc = nn.Linear(latent_dim, c2 * 7 * 7)
        self.c2 = c2
        self.deconv1 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(c1, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        h = F.relu(self.fc(z))
        h = h.view(-1, self.c2, 7, 7)
        h = F.relu(self.deconv1(h))
        return torch.sigmoid(self.deconv2(h))


class VAE(nn.Module):
    """#29 VAE 모델 — checkpoint 호환을 위한 인라인 정의."""
    def __init__(self, latent_dim=VAE_LATENT_DIM, hidden_channels=VAE_HIDDEN_CHANNELS):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim, hidden_channels)
        self.decoder = VAEDecoder(latent_dim, hidden_channels)

    @torch.no_grad()
    def sample(self, n_samples, device=DEVICE):
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decoder(z)


def load_vae_from_29(device=DEVICE):
    """#29 VAE checkpoint를 로드해서 06번 비교 시각화에 사용."""
    vae_ckpt_path = ROOT.parent / "29. vae-from-scratch" / "data" / "checkpoint.pt"
    if not vae_ckpt_path.exists():
        return None
    try:
        ckpt = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
        vae = VAE().to(device)
        vae.load_state_dict(ckpt["model"])
        vae.eval()
        return vae
    except Exception as e:
        print(f"[warn] VAE checkpoint 로드 실패: {e}", flush=True)
        return None


# ───────────────────────────────────────────────────────────
# 7. 학습 루프
# ───────────────────────────────────────────────────────────
def train_loop(G, D, train_loader):
    G.train()
    D.train()
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
    bce = nn.BCEWithLogitsLoss()

    history = []
    snapshots = {}

    # Fixed z: 학습 진행에 따른 변화를 일관되게 모니터링
    fixed_z = torch.randn(FIXED_Z_SAMPLES, Z_DIM, device=DEVICE)

    print("[train] start", flush=True)
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        g_loss_sum, d_loss_sum = 0.0, 0.0
        d_real_correct, d_fake_correct = 0, 0
        n_samples = 0

        for real_imgs, _ in train_loader:
            real_imgs = real_imgs.to(DEVICE)
            batch = real_imgs.size(0)

            # === Train Discriminator ===
            opt_D.zero_grad()

            # (1) Real images → D should output 1
            real_labels = torch.ones(batch, 1, device=DEVICE)
            d_real_logits = D(real_imgs)
            d_real_loss = bce(d_real_logits, real_labels)

            # (2) Fake images → D should output 0
            z = torch.randn(batch, Z_DIM, device=DEVICE)
            fake_imgs = G(z)
            fake_labels = torch.zeros(batch, 1, device=DEVICE)
            d_fake_logits = D(fake_imgs.detach())
            d_fake_loss = bce(d_fake_logits, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            opt_D.step()

            # === Train Generator ===
            opt_G.zero_grad()
            # G는 D가 fake를 real(=1)로 속도록 학습 (non-saturating loss)
            d_fake_logits_for_g = D(fake_imgs)
            g_loss = bce(d_fake_logits_for_g, real_labels)
            g_loss.backward()
            opt_G.step()

            # Track metrics
            g_loss_sum += g_loss.item() * batch
            d_loss_sum += d_loss.item() * batch
            with torch.no_grad():
                d_real_pred = (torch.sigmoid(d_real_logits) > 0.5).float()
                d_fake_pred = (torch.sigmoid(d_fake_logits) > 0.5).float()
                d_real_correct += d_real_pred.sum().item()
                d_fake_correct += (1 - d_fake_pred).sum().item()
            n_samples += batch

        g_loss_avg = g_loss_sum / n_samples
        d_loss_avg = d_loss_sum / n_samples
        d_acc_real = d_real_correct / n_samples
        d_acc_fake = d_fake_correct / n_samples
        d_acc_total = (d_real_correct + d_fake_correct) / (2 * n_samples)

        history.append({
            "epoch": epoch,
            "g_loss": g_loss_avg,
            "d_loss": d_loss_avg,
            "d_acc_real": d_acc_real,
            "d_acc_fake": d_acc_fake,
            "d_acc_total": d_acc_total,
        })

        # Snapshot for training progression viz
        if epoch in SNAPSHOT_EPOCHS:
            G.eval()
            with torch.no_grad():
                snapshots[epoch] = G(fixed_z).cpu()
            G.train()

        elapsed = time.time() - t0
        print(f"[epoch {epoch:3d}] G_loss={g_loss_avg:.4f}  D_loss={d_loss_avg:.4f}  "
              f"D_acc(real)={d_acc_real:.3f}  D_acc(fake)={d_acc_fake:.3f}  "
              f"D_acc(total)={d_acc_total:.3f}  ({elapsed:.1f}s)", flush=True)

    return history, snapshots, fixed_z


# ───────────────────────────────────────────────────────────
# 8. 시각화
# ───────────────────────────────────────────────────────────

def _to_display(img_tensor):
    """Tanh output [-1, 1] → display range [0, 1]."""
    return ((img_tensor + 1) / 2).clamp(0, 1)


def plot_dataset_overview(train_set):
    """01: FashionMNIST 개요 — 10개 클래스 + 분포 + 통계."""
    fig = plt.figure(figsize=(16, 8))

    samples = {}
    for img, lbl in train_set:
        lbl = int(lbl)
        if lbl not in samples:
            samples[lbl] = img
        if len(samples) == 10:
            break

    for i in range(10):
        row, col = i // 5, i % 5
        ax = plt.subplot2grid((3, 5), (row, col))
        # FashionMNIST는 [-1, 1] 정규화 후 이미지로 다시 [0, 1] 변환
        img = _to_display(samples[i]).squeeze().numpy()
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{i}: {CLASS_NAMES[i]}", fontsize=11, fontweight="bold")
        ax.axis("off")

    # 클래스 분포
    ax_dist = plt.subplot2grid((3, 5), (2, 0), colspan=3)
    targets = train_set.targets.numpy() if hasattr(train_set.targets, "numpy") else np.array(train_set.targets)
    counts = np.bincount(targets, minlength=10)
    ax_dist.bar(range(10), counts, color="steelblue")
    ax_dist.set_xticks(range(10))
    ax_dist.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=10)
    ax_dist.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax_dist.set_title("Train Set Class Distribution", fontsize=13, fontweight="bold")
    ax_dist.grid(True, alpha=0.3, axis="y")

    # 통계
    ax_stats = plt.subplot2grid((3, 5), (2, 3), colspan=2)
    ax_stats.axis("off")
    stats_lines = [
        f"Image size      : 28 × 28 (grayscale)",
        f"Train samples   : {len(train_set):,}",
        f"Number of classes: 10",
        f"Pixel range     : [-1, 1]  (Tanh-matched)",
        f"Latent z dim    : {Z_DIM}",
    ]
    ax_stats.text(0.0, 0.85, "\n".join(stats_lines),
                  fontsize=12, family="monospace", fontweight="bold", va="top")
    ax_stats.set_title("Dataset Statistics", loc="left", fontsize=13, fontweight="bold")

    plt.suptitle("FashionMNIST Dataset Overview", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_dataset_overview.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 01_dataset_overview.png")


def plot_training_curve(history):
    """02: G/D Loss + D Accuracy (0.5 수렴 검증)."""
    epochs = [h["epoch"] for h in history]
    g_loss = [h["g_loss"] for h in history]
    d_loss = [h["d_loss"] for h in history]
    d_acc_real = [h["d_acc_real"] for h in history]
    d_acc_fake = [h["d_acc_fake"] for h in history]
    d_acc_total = [h["d_acc_total"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (좌) G loss vs D loss
    ax = axes[0]
    ax.plot(epochs, g_loss, label="Generator loss", color="steelblue",
            linewidth=2.2, marker="o", markersize=4)
    ax.plot(epochs, d_loss, label="Discriminator loss", color="crimson",
            linewidth=2.2, marker="s", markersize=4)
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title("Generator vs Discriminator Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    # (우) D Accuracy + Equilibrium line
    ax = axes[1]
    ax.plot(epochs, d_acc_real, label="D Acc (real images)", color="darkgreen",
            linewidth=2.0, marker="o", markersize=4)
    ax.plot(epochs, d_acc_fake, label="D Acc (fake images)", color="purple",
            linewidth=2.0, marker="s", markersize=4)
    ax.plot(epochs, d_acc_total, label="D Acc (total)", color="darkorange",
            linewidth=2.6, marker="^", markersize=5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.8, alpha=0.7,
               label="Equilibrium (0.5)")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_title("Discriminator Accuracy  —  0.5 means D is fully fooled",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    plt.suptitle("GAN Training Dynamics  —  Adversarial Balance",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_training_curve.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 02_training_curve.png")


def plot_training_progression(snapshots):
    """03: 같은 z, 다른 epoch — 학습이 진행될수록 z가 어떻게 의미 있는 이미지로 변하는지."""
    epochs = sorted(snapshots.keys())
    n_samples = next(iter(snapshots.values())).shape[0]

    fig, axes = plt.subplots(len(epochs), n_samples,
                             figsize=(n_samples * 1.5, len(epochs) * 1.6))

    for i, epoch in enumerate(epochs):
        for j in range(n_samples):
            ax = axes[i, j]
            img = _to_display(snapshots[epoch][j]).squeeze().numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(f"Epoch {epoch}", fontsize=12, fontweight="bold",
                              rotation=0, ha="right", va="center", labelpad=20)

    # 컬럼 라벨
    for j in range(n_samples):
        axes[0, j].set_title(f"z_{j+1}", fontsize=10, fontweight="bold", color="#555555")

    plt.suptitle("Training Progression  —  Same z, Different Epochs",
                 fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "03_training_progression.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 03_training_progression.png")


def plot_diversity_and_d_score(G, D, train_set):
    """04: 64개 생성 샘플(다양성 / Mode Collapse 체크) + D Score 분포 (real vs fake)."""
    G.eval()
    D.eval()

    # (a) 64 generated samples
    with torch.no_grad():
        z = torch.randn(64, Z_DIM, device=DEVICE)
        fake_imgs = G(z).cpu()

    # (b) D scores
    real_loader = DataLoader(train_set, batch_size=1000, shuffle=True, num_workers=0)
    real_batch, _ = next(iter(real_loader))
    real_batch = real_batch.to(DEVICE)
    with torch.no_grad():
        real_scores = torch.sigmoid(D(real_batch)).cpu().numpy().flatten()
        z_big = torch.randn(1000, Z_DIM, device=DEVICE)
        fake_big = G(z_big)
        fake_scores = torch.sigmoid(D(fake_big)).cpu().numpy().flatten()

    fig = plt.figure(figsize=(16, 8))

    # (a) Samples grid 8×8
    gs_left = fig.add_gridspec(8, 8, left=0.02, right=0.50, top=0.91, bottom=0.05,
                                hspace=0.04, wspace=0.04)
    for i in range(64):
        ax = fig.add_subplot(gs_left[i // 8, i % 8])
        img = _to_display(fake_imgs[i]).squeeze().numpy()
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.text(0.26, 0.94, "(a) 64 Generated Samples  —  Mode Diversity Check",
             fontsize=14, fontweight="bold", ha="center")

    # (b) D Score histogram
    gs_right = fig.add_gridspec(1, 1, left=0.59, right=0.97, top=0.82, bottom=0.10)
    ax = fig.add_subplot(gs_right[0, 0])
    ax.hist(real_scores, bins=30, alpha=0.65, color="darkgreen",
            label=f"Real (mean={real_scores.mean():.3f})",
            edgecolor="black", linewidth=0.5)
    ax.hist(fake_scores, bins=30, alpha=0.65, color="crimson",
            label=f"Generated (mean={fake_scores.mean():.3f})",
            edgecolor="black", linewidth=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.8, alpha=0.7,
               label="Decision threshold (0.5)")
    ax.set_xlabel("D Score (sigmoid output)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count (n=1000 each)", fontsize=12, fontweight="bold")
    ax.set_title("(b) D Score Distribution  —  Real vs Generated",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)

    plt.suptitle("Mode Diversity & Adversarial Balance",
                 fontsize=16, fontweight="bold", y=0.99)
    plt.savefig(RESULTS_DIR / "04_diversity_and_d_score.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 04_diversity_and_d_score.png")


def plot_latent_walk(G, n_pairs=5, n_steps=10):
    """05: z-space interpolation — 5쌍의 z를 10단계로 보간."""
    G.eval()

    fig, axes = plt.subplots(n_pairs, n_steps,
                             figsize=(n_steps * 1.3, n_pairs * 1.5))

    with torch.no_grad():
        for i in range(n_pairs):
            z_start = torch.randn(1, Z_DIM, device=DEVICE)
            z_end = torch.randn(1, Z_DIM, device=DEVICE)
            alphas = torch.linspace(0, 1, n_steps, device=DEVICE).view(-1, 1)
            z_interp = (1 - alphas) * z_start + alphas * z_end
            imgs = G(z_interp).cpu()

            for j in range(n_steps):
                ax = axes[i, j]
                img = _to_display(imgs[j]).squeeze().numpy()
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
            axes[i, 0].set_ylabel(f"Pair {i+1}", fontsize=11, fontweight="bold",
                                  rotation=0, ha="right", va="center", labelpad=15)

    # α labels
    for j, alpha in enumerate(np.linspace(0, 1, n_steps)):
        axes[0, j].set_title(f"α={alpha:.2f}", fontsize=10, fontweight="bold", color="#555555")

    plt.suptitle("Latent Space Interpolation  —  Smooth Manifold Check",
                 fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "05_latent_walk.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 05_latent_walk.png")


def plot_vae_vs_gan_comparison(G):
    """06: VAE vs GAN — 같은 조건에서 두 모델의 출력 비교 (blurry vs sharp)."""
    G.eval()

    n_samples = 8
    # GAN samples
    with torch.no_grad():
        z_gan = torch.randn(n_samples, Z_DIM, device=DEVICE)
        gan_samples = G(z_gan).cpu()

    # VAE samples (#29 checkpoint)
    vae = load_vae_from_29(DEVICE)
    if vae is None:
        print("[warn] VAE checkpoint를 찾을 수 없어 06번을 GAN-only로 저장합니다.", flush=True)
        # Fallback: GAN 16개만 표시
        fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
        with torch.no_grad():
            z_more = torch.randn(16, Z_DIM, device=DEVICE)
            extra = G(z_more).cpu()
        for i in range(16):
            ax = axes[i // 8, i % 8]
            img = _to_display(extra[i]).squeeze().numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        plt.suptitle("GAN Generated Samples (VAE checkpoint not found)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "06_vae_vs_gan_comparison.png", dpi=120, bbox_inches="tight")
        plt.close()
        print("[viz] saved 06_vae_vs_gan_comparison.png  (VAE-less fallback)")
        return

    with torch.no_grad():
        vae_samples = vae.sample(n_samples, device=DEVICE).cpu()
    # VAE는 sigmoid output [0, 1], display 그대로

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.7, 4.5))

    for j in range(n_samples):
        # 위: VAE
        ax = axes[0, j]
        img = vae_samples[j].squeeze().numpy()
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        # 아래: GAN
        ax = axes[1, j]
        img = _to_display(gan_samples[j]).squeeze().numpy()
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row labels
    axes[0, 0].set_ylabel("VAE\n(blurry)", fontsize=13, fontweight="bold",
                          rotation=0, ha="right", va="center", labelpad=25, color="#1f4e8f")
    axes[1, 0].set_ylabel("GAN\n(sharp)", fontsize=13, fontweight="bold",
                          rotation=0, ha="right", va="center", labelpad=25, color="#a02020")

    plt.suptitle("VAE vs GAN  —  Same Domain, Different Strengths",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "06_vae_vs_gan_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 06_vae_vs_gan_comparison.png")


# ───────────────────────────────────────────────────────────
# 9. 메인 파이프라인 (체크포인트 save/load 지원)
# ───────────────────────────────────────────────────────────
CHECKPOINT_PATH = DATA_DIR / "checkpoint.pt"


def main():
    """
    실행 옵션:
        python src/main.py             # checkpoint 있으면 load, 없으면 학습 후 저장
        python src/main.py --retrain   # 강제 재학습
        python src/main.py --viz-only  # checkpoint만 load, 학습 스킵
    """
    force_retrain = "--retrain" in sys.argv
    viz_only = "--viz-only" in sys.argv

    print(f"[info] device      : {DEVICE}")
    print(f"[info] z dim       : {Z_DIM}")
    print(f"[info] batch size  : {BATCH_SIZE}")
    print(f"[info] epochs      : {EPOCHS}", flush=True)

    train_loader, train_set = get_dataloader()
    print(f"[info] train size  : {len(train_set):,}", flush=True)

    # 시각화 01: 데이터셋 개요 (학습 무관)
    plot_dataset_overview(train_set)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    print(f"[info] G params    : {G.num_params() / 1e6:.2f} M")
    print(f"[info] D params    : {D.num_params() / 1e6:.2f} M", flush=True)

    if CHECKPOINT_PATH.exists() and not force_retrain:
        print(f"[load] checkpoint found at {CHECKPOINT_PATH.name}, skipping training", flush=True)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        history = ckpt["history"]
        snapshots = ckpt["snapshots"]
    else:
        if viz_only:
            raise FileNotFoundError(f"--viz-only 모드인데 checkpoint가 없습니다: {CHECKPOINT_PATH}")
        history, snapshots, _ = train_loop(G, D, train_loader)
        torch.save(
            {
                "G": G.state_dict(),
                "D": D.state_dict(),
                "history": history,
                "snapshots": snapshots,
            },
            CHECKPOINT_PATH,
        )
        print(f"[save] checkpoint saved to {CHECKPOINT_PATH.name}", flush=True)

    # 시각화 02~06
    plot_training_curve(history)
    plot_training_progression(snapshots)
    plot_diversity_and_d_score(G, D, train_set)
    plot_latent_walk(G)
    plot_vae_vs_gan_comparison(G)

    print("[done] all visualizations saved to results/")


if __name__ == "__main__":
    main()
