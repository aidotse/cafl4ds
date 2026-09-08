"""Scout: MAE inpainting difficulty (reconstruction loss) across candidate phase-B domains.

P0.3.10/Q2 diagnostic. In MAE the reconstruction-loss *magnitude* sets the gradient magnitude,
hence the encoder-update pressure a phase-B domain applies (the P0.4.1 finding: recon loss reads
inpaintability, not distributional novelty). A far domain that is *easy* to inpaint (EuroSAT, blurred
64px satellite tiles) applies weak pressure and looks like "resistance" for the wrong reason. To
isolate distributional distance from update pressure, phase B must be a far domain whose recon loss
matches task A's. This scout measures that difficulty at the *phase-B-start* condition — a decoder
warmed on task A (frozen encoder, as in the harness decoder-first phase A), then recon read on each
domain at that fixed decoder state (norm_pix_loss=True, the training setting; 8 mask draws averaged).

Run in the Gaudi container (reuses the harness mounts):
    ./scripts/run_gaudi_dev.sh -m /mnt/cifar100 -m /mnt/imagenette -m /home/mauricio/data \
        gaudi-env-cafl4ds:latest 0 python scripts/scout_recon_difficulty.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from loguru import logger
from torch import optim
from torchvision import transforms
from torchvision.datasets import DTD

from cafl4ds.data.sources import EuroSATSource, ImagenetteSource
from cafl4ds.models.vit_mae_pretrained import MAEPretrainedViTEncoder
from cafl4ds.ssl.factory import build_mae
from cafl4ds.ssl.mae import MAE

CKPT = "/mnt/cifar100/mae_pretrain_vit_base.pth"
WARM_EPOCHS = 12
BS = 64
MASK_DRAWS = 8


def _load_dtd(root: str, img_size: int, max_imgs: int) -> torch.Tensor:
    ds = DTD(root=root, split="train", download=False)
    resize = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    imgs = [resize(ds[i][0]) for i in range(min(max_imgs, len(ds)))]
    return torch.stack(imgs)


def _recon(method: MAE, imgs: torch.Tensor, device: torch.device, side: int) -> float:
    was = method.training
    method.eval()
    orig = method.augment
    method.augment = torch.nn.Identity()
    try:
        with torch.no_grad():
            moved = imgs.to(device)
            if moved.shape[-1] != side or moved.shape[-2] != side:
                moved = F.interpolate(moved, size=side, mode="bilinear", align_corners=False)
            draws = [float(method.per_sample_loss(moved).mean().item()) for _ in range(MASK_DRAWS)]
            return sum(draws) / len(draws)
    finally:
        method.augment = orig
        method.train(was)


def main() -> None:
    """Warm the decoder on task A, then report norm-pix recon loss on each candidate phase-B domain."""
    device = torch.device("hpu")
    encoder = MAEPretrainedViTEncoder(
        state_dict_path=CKPT, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0
    )
    method = build_mae(
        encoder,
        mask_ratio=0.75,
        decoder_dim=64,
        decoder_depth=2,
        decoder_heads=4,
        decoder_mlp_ratio=2.0,
        norm_pix_loss=True,
    ).to(device)
    side = int(round(encoder.num_patches**0.5)) * encoder.patch_size
    logger.info(f"encoder native side = {side}px")

    img_a, _ = ImagenetteSource(root="/mnt/imagenette/imagenette2-320", img_size=224, max_per_class=300).load()
    img_e, _ = EuroSATSource(root="/home/mauricio/data", img_size=64, max_per_class=300).load()
    img_d = _load_dtd("/home/mauricio/data/dtd", 224, 600)
    logger.info(f"loaded imagenette={img_a.shape[0]} eurosat={img_e.shape[0]} dtd={img_d.shape[0]}")

    # Warm the decoder on task A (freeze the encoder) — the phase-B-start condition.
    warm, held = img_a[:-120], img_a[-120:]
    for p in method.encoder.parameters():
        p.requires_grad = False
    opt = optim.AdamW([p for p in method.decoder.parameters() if p.requires_grad], lr=1e-4)
    method.train()
    for ep in range(WARM_EPOCHS):
        perm = torch.randperm(warm.shape[0])
        losses = []
        for i in range(0, warm.shape[0], BS):
            batch = warm[perm[i : i + BS]].to(device)
            if batch.shape[0] < 2:
                continue
            opt.zero_grad()
            loss = method.training_step(batch)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        logger.info(f"decoder-warm epoch {ep + 1}/{WARM_EPOCHS}: mean loss {sum(losses) / len(losses):.4f}")

    r_a = _recon(method, held, device, side)
    r_e = _recon(method, img_e, device, side)
    r_d = _recon(method, img_d, device, side)
    logger.info("=== inpainting difficulty (norm_pix recon loss, task-A-warmed decoder) ===")
    logger.info(f"  task-A Imagenette (held-out) : {r_a:.4f}   <- the match target")
    logger.info(f"  EuroSAT (current far domain) : {r_e:.4f}   (ratio to task-A {r_e / r_a:.2f}x)")
    logger.info(f"  DTD textures (candidate)     : {r_d:.4f}   (ratio to task-A {r_d / r_a:.2f}x)")


if __name__ == "__main__":
    main()
