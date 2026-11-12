"""P1.2.5 Part E's second corroborating experiment: does BDD's sameness flatten DINO's centred target?

The released DINO teacher (backbone + head, from the full checkpoint) reads the two global crops of N frames from each
image set; the set's centre is the mean of those logits (the value DINO's running centre converges to). Per set:

- the teacher target's mean entropy (nats; uniform = ln K) with the set's own centre, with none, and with BDD's centre;
- how many of the K prototypes the set's views pick as their top choice;
- how alike the frames look to the teacher: the mean pairwise cosine of the backbone's ``cls`` features.

Forward passes only. Run in the Gaudi container from the repo root, with the share mounted:
    ./scripts/run_gaudi_dev.sh -m /mnt/pr_2018_scaleout_workdir/cafl4ds/data gaudi-env-cafl4ds:latest 0 \
        python scripts/dino_centring_check.py --device hpu --out outputs/p125/centring/centring.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf

from cafl4ds.data.attributes import as_float
from cafl4ds.data.sources import ImagenetteSource
from cafl4ds.models.vit_mae_pretrained import MAEPretrainedViTEncoder
from cafl4ds.ssl.dino import DINO, build_dino

DATA = "/mnt/pr_2018_scaleout_workdir/cafl4ds/data"
CACHE = "outputs/cache"
# The decoded 224 px caches the P1.2.5 drives built (BDD train: the canary-scene pool; COCO / Places365: Part D's
# controls, driving categories excluded).
CACHES = {
    "bdd": f"{CACHE}/bdd_train_224px_uint8_263817c6124b6da3.pt",
    "coco": f"{CACHE}/coco_224px_uint8_05cd31b72aa6946d.pt",
    "places365": f"{CACHE}/places365_224px_uint8_fb981119d5e85d50.pt",
}


def target_entropy(logits: torch.Tensor, center: torch.Tensor | None, temp: float) -> torch.Tensor:
    """Per-row entropy (nats) of DINO's teacher target ``softmax((logits - center) / temp)``."""
    shifted = logits if center is None else logits - center
    logp = F.log_softmax(shifted / temp, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def mean_pairwise_cosine(features: torch.Tensor) -> float:
    """Mean cosine similarity over all distinct pairs of rows."""
    z = F.normalize(features.double(), dim=-1)
    n = z.shape[0]
    total = (z.sum(dim=0) ** 2).sum() - n  # sum over i != j of z_i . z_j
    return float(total / (n * (n - 1)))


def set_stats(logits: torch.Tensor, cls: torch.Tensor, ref_center: torch.Tensor, temp: float) -> dict[str, float]:
    """The read for one image set: entropies under three centres, prototypes in use, frame similarity."""
    center = logits.mean(dim=0, keepdim=True)
    return {
        "entropy_own_center": float(target_entropy(logits, center, temp).mean()),
        "entropy_no_center": float(target_entropy(logits, None, temp).mean()),
        "entropy_bdd_center": float(target_entropy(logits, ref_center, temp).mean()),
        "uniform_entropy": math.log(logits.shape[1]),
        "top_prototypes_used": int((logits - center).argmax(dim=-1).unique().numel()),
        "top_prototypes_used_no_center": int(logits.argmax(dim=-1).unique().numel()),
        "cls_mean_pairwise_cosine": mean_pairwise_cosine(cls),
    }


def teacher_read(method: DINO, images: torch.Tensor, batch: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Teacher logits on both global crops ``[2N, K]`` and the backbone ``cls`` on the full frame ``[N, D]``."""
    logits, cls = [], []
    with torch.no_grad():
        for imgs in images.split(batch):
            x = as_float(imgs).to(device)
            for t in method.crops.globals:  # one draw per batch, as in training
                logits.append(method.teacher_head(method._cls(method.encoder, t(x))).float().cpu())
            cls.append(method._cls(method.encoder, x).float().cpu())
    return torch.cat(logits), torch.cat(cls)


def load_sets(n: int, seed: int) -> dict[str, torch.Tensor]:
    """``n`` frames per set, drawn by ``seed``, uint8 or float ``[n, 3, 224, 224]``."""
    gen = torch.Generator().manual_seed(seed)
    sets: dict[str, torch.Tensor] = {}
    for name, path in CACHES.items():
        images = torch.load(path, weights_only=False, mmap=True).images
        keep = torch.randperm(images.shape[0], generator=gen)[:n].sort().values
        sets[name] = images[keep].clone()
        logger.info(f"{name}: {n} of {images.shape[0]} frames")
    nette, _ = ImagenetteSource(f"{DATA}/imagenette/imagenette2-320", split="train", img_size=224).load()
    sets["imagenette"] = nette[torch.randperm(nette.shape[0], generator=gen)[:n].sort().values]
    logger.info(f"imagenette: {n} of {nette.shape[0]} frames")
    return sets


def main() -> None:
    """Read the four image sets and write the per-set statistics as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/p125/centring/centring.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    cfg = OmegaConf.load("cafl4ds/configs/backbone/dino.yaml").ssl
    kwargs = {k: v for k, v in OmegaConf.to_container(cfg).items() if k != "_target_"}
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}
    encoder = MAEPretrainedViTEncoder(state_dict_path=f"{DATA}/weights/dino_vitbase16_pretrain.pth", img_size=224)
    method = build_dino(encoder, **kwargs).to(device).eval()

    reads = {
        name: teacher_read(method, imgs, args.batch, device) for name, imgs in load_sets(args.n, args.seed).items()
    }
    bdd_center = reads["bdd"][0].mean(dim=0, keepdim=True)
    stats = {name: set_stats(lg, cls, bdd_center, method.teacher_temp) for name, (lg, cls) in reads.items()}
    for name, s in stats.items():
        logger.info(f"{name}: " + ", ".join(f"{k}={v:.4g}" for k, v in s.items()))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"n": args.n, "seed": args.seed, "teacher_temp": method.teacher_temp, "sets": stats}, indent=2)
    )
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
