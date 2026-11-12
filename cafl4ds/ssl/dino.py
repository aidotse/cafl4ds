"""DINO SSL method — self-distillation with no labels (Caron et al. 2021), resumed from its own checkpoint (P1.2.5).

P1.2's joint-embedding arm so far is a hybrid: MAE's weights under SimSiam's objective. P1.2.5 continues the
DINO-pretrained ViT-B/16 with DINO's *own* objective, so the joint-embedding result carries no MAE and no
objective switch. The method mirrors the official recipe:

* a **student** (backbone + head) trained by gradient, and a **teacher** of the same shape that is the student's
  exponential moving average (momentum ramped 0.996 → 1 over the drive, DINO's cosine schedule);
* **multi-crop** — two global crops (224 px, scale 0.4–1) seen by both, and local crops (96 px, scale 0.05–0.4)
  seen by the student only; each student crop is pulled toward the teacher's distribution on the *other* global
  crops (cross-entropy over the head's prototypes);
* **centering** (a running mean subtracted from the teacher's logits, momentum 0.9) and **sharpening** (a teacher
  temperature below the student's) — the two mechanisms that, together, keep DINO from collapsing.

**The centre at step 0.** The released checkpoint holds no centre, so :meth:`DINO.prime` estimates it before the
drive: the teacher's mean logits on the global crops of the well's frames, with no gradient. Nothing the probes read
moves, so step 0 is still the frozen floor.

**What is read.** The teacher is the encoder DINO evaluates, so it is this method's ``encoder`` — the one every
probe, health read and checkpoint read uses; the student is ``student``. At step 0 the teacher is the released
DINO backbone, so the drive starts bit-identical to the frozen floor (B5-DINO).

**The collapse switches** (Phase 0's control for any new joint-embedding method, P1.2.5 Part E). Each turns one
mechanism off and leaves the other on, which DINO's paper shows collapses the model in a known direction:
``centering=False`` (sharpening alone: one dimension dominates) and ``sharpening=False`` (the teacher at the
student's temperature, centering alone: a uniform output). The method's ``name`` records the switch.

**A harness simplification**, as for SimSiam (:mod:`cafl4ds.ssl.augment`): each crop's augmentation draws its
parameters once per batch, so the images of a batch share a crop box and a jitter while every crop differs.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias
from loguru import logger
from torch import nn
from torchvision.transforms import v2

from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.base import SSLMethod


class _WeightNormLinear(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """A bias-free weight-normalized linear layer, keyed as the official head's (``weight_g`` / ``weight_v``).

    DINO's last layer is ``nn.utils.weight_norm(nn.Linear(...))``; written out here so the official checkpoint
    loads by name and no deprecated parametrization is needed. With ``norm_last_layer`` the gain is fixed at 1.
    """

    def __init__(self, in_dim: int, out_dim: int, fix_gain: bool) -> None:
        super().__init__()
        self.weight_v = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.trunc_normal_(self.weight_v, std=0.02)
        self.weight_g = nn.Parameter(torch.ones(out_dim, 1), requires_grad=not fix_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_g * self.weight_v / self.weight_v.norm(dim=1, keepdim=True)
        return F.linear(x, weight)


class DINOHead(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """DINO's projection head: an MLP to a bottleneck, L2-normalized, then a weight-normed layer to the prototypes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
        use_bn: bool = False,
        norm_last_layer: bool = True,
    ) -> None:
        """Build the head with the official layer names (``mlp.<i>``, ``last_layer.weight_g`` / ``weight_v``)."""
        super().__init__()
        layers: list[nn.Module] = []
        dims = [in_dim, *[hidden_dim] * (nlayers - 1)]
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(a, b), *([nn.BatchNorm1d(b)] if use_bn else []), nn.GELU()]
        layers.append(nn.Linear(dims[-1], bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.last_layer = _WeightNormLinear(bottleneck_dim, out_dim, fix_gain=norm_last_layer)

    def bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """The MLP's output ``[B, bottleneck_dim]`` (before the L2 normalization) — the health monitor's ``proj``."""
        return self.mlp(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prototype logits ``[B, out_dim]``."""
        return self.last_layer(F.normalize(self.mlp(x), dim=-1))


def _crop_augment(size: int, scale: tuple[float, float], blur: float, solarize: float) -> v2.Transform:
    """One DINO crop's augmentation (flip, colour jitter, grayscale, blur, solarization), on ``[0, 1]`` floats."""
    return v2.Compose(
        [
            v2.RandomResizedCrop(size=size, scale=scale, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=2 * (size // 20) + 1, sigma=(0.1, 2.0))], p=blur),
            v2.RandomSolarize(threshold=0.5, p=solarize),
        ]
    )


class MultiCrop(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """DINO's multi-crop: two global crops (blur 1.0 / 0.1, solarize 0 / 0.2) and ``n_local`` local crops (blur 0.5)."""

    def __init__(
        self,
        global_size: int,
        local_size: int,
        n_local: int,
        global_scale: tuple[float, float] = (0.4, 1.0),
        local_scale: tuple[float, float] = (0.05, 0.4),
    ) -> None:
        """Two global crops at ``global_size`` and ``n_local`` local crops at ``local_size``, at their scales."""
        super().__init__()
        self.globals = [
            _crop_augment(global_size, global_scale, blur=1.0, solarize=0.0),
            _crop_augment(global_size, global_scale, blur=0.1, solarize=0.2),
        ]
        self.local = _crop_augment(local_size, local_scale, blur=0.5, solarize=0.0)
        self.n_local = n_local

    def forward(self, imgs: torch.Tensor) -> list[torch.Tensor]:
        """The global crops first, then the local ones."""
        return [t(imgs) for t in self.globals] + [self.local(imgs) for _ in range(self.n_local)]


def momentum_at(step: int, total: int, base: float) -> float:
    """DINO's teacher momentum after ``step`` of ``total`` steps: a cosine ramp from ``base`` to 1."""
    if total <= 1:
        return base
    return 1.0 - (1.0 - base) * (math.cos(math.pi * min(step, total) / total) + 1.0) / 2.0


class DINO(SSLMethod):
    """DINO over a ViT backbone: the student trains, the moving-average teacher is the encoder read (see module)."""

    def __init__(
        self,
        encoder: TinyViTEncoder,
        student: TinyViTEncoder,
        student_head: DINOHead,
        teacher_head: DINOHead,
        crops: MultiCrop,
        *,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        teacher_momentum: float = 0.996,
        centering: bool = True,
        sharpening: bool = True,
        center: torch.Tensor | None = None,
    ) -> None:
        """Assemble the method.

        Args:
            encoder: The teacher backbone — the encoder read, never trained by gradient.
            student: The student backbone (trained).
            student_head: The student's projection head (trained).
            teacher_head: The teacher's head (the student head's moving average).
            crops: The multi-crop augmentation.
            student_temp: The student's softmax temperature.
            teacher_temp: The teacher's (sharpening: below the student's).
            center_momentum: The running centre's momentum.
            teacher_momentum: The teacher's starting momentum (ramped to 1 over :meth:`set_horizon`'s steps).
            centering: Subtract the running centre from the teacher's logits (off: collapse switch).
            sharpening: Use ``teacher_temp`` (off: the teacher at ``student_temp`` — collapse switch).
            center: The starting centre ``[1, out_dim]`` (the checkpoint's), or ``None``: zeros until :meth:`prime`.
        """
        super().__init__(encoder)
        self.student = student
        self.student_head = student_head
        self.teacher_head = teacher_head
        self.crops = crops
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp if sharpening else student_temp
        self.center_momentum = center_momentum
        self.teacher_momentum = teacher_momentum
        self.centering = centering
        self.sharpening = sharpening
        out_dim = int(teacher_head.last_layer.weight_v.shape[0])
        self.register_buffer("center", center.clone() if center is not None else torch.zeros(1, out_dim))
        self._center_known = center is not None
        self._step = 0
        self._total = 0
        for param in [*self.encoder.parameters(), *self.teacher_head.parameters()]:
            param.requires_grad_(False)

    @property
    def name(self) -> str:
        """``dino``, or ``dino_nocenter`` / ``dino_nosharpen`` with a collapse switch thrown."""
        if not self.centering:
            return "dino_nocenter"
        return "dino" if self.sharpening else "dino_nosharpen"

    def set_horizon(self, total_steps: int) -> None:
        """Ramp the teacher momentum to 1 over ``total_steps`` (restarting the ramp)."""
        self._step, self._total = 0, int(total_steps)

    @torch.no_grad()  # type: ignore[misc]  # untyped decorator without torch stubs (mypy hook env)
    def prime(self, batches: Iterable[torch.Tensor]) -> None:
        """Estimate a centre the checkpoint lacks: the teacher's mean logits on the global crops of ``batches``.

        A no-op when the centre is known or centering is off. Run on the method's device.
        """
        if self._center_known or not self.centering:
            return
        total, count = torch.zeros_like(self.center), 0
        for imgs in batches:
            views = [t(imgs.to(self.center.device)) for t in self.crops.globals]
            logits = self.teacher_head(self._cls(self.encoder, torch.cat(views)))
            total += logits.sum(dim=0, keepdim=True)
            count += logits.shape[0]
        if count == 0:
            logger.warning("no frames to estimate DINO's centre from: it stays at zero")
            return
        self.center.copy_(total / count)
        self._center_known = True
        logger.info(f"DINO centre estimated over {count // 2} frames (norm {float(self.center.norm()):.3f})")

    @staticmethod
    def _cls(backbone: TinyViTEncoder, view: torch.Tensor) -> torch.Tensor:
        """The backbone's ``cls`` token on a crop at its own resolution (local crops on interpolated positions)."""
        tokens, _, _ = backbone.forward_encoder(view, resize=False)
        return tokens[:, 0]

    def _student_logits(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """Student logits per crop; consecutive crops of one size share a forward, as DINO's MultiCropWrapper."""
        out: list[torch.Tensor] = []
        for _, group in itertools.groupby(views, key=lambda v: v.shape[-1]):
            crops = list(group)
            out += list(self.student_head(self._cls(self.student, torch.cat(crops))).chunk(len(crops)))
        return out

    def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
        """DINO's loss: each student crop against the teacher's distribution on every *other* global crop.

        Updates the running centre from the teacher's raw logits (as DINO does, inside the loss).
        """
        views = self.crops(imgs)
        with torch.no_grad():
            teacher_logits = self.teacher_head(self._cls(self.encoder, torch.cat(views[:2])))
        shifted = teacher_logits - self.center if self.centering else teacher_logits
        targets = F.softmax(shifted / self.teacher_temp, dim=-1).chunk(2)
        student = self._student_logits(views)
        total, terms = torch.zeros((), device=teacher_logits.device), 0
        for iq, q in enumerate(targets):
            for v, logits in enumerate(student):
                if v == iq:
                    continue
                total = total + torch.sum(-q * F.log_softmax(logits / self.student_temp, dim=-1), dim=-1).mean()
                terms += 1
        if self.centering:
            with torch.no_grad():
                batch_center = teacher_logits.mean(dim=0, keepdim=True)
                self.center.mul_(self.center_momentum).add_(batch_center, alpha=1.0 - self.center_momentum)
        return total / terms

    @torch.no_grad()  # type: ignore[misc]  # untyped decorator without torch stubs (mypy hook env)
    def after_step(self) -> None:
        """The teacher (backbone and head) moves toward the student by the scheduled momentum."""
        self._step += 1
        m = momentum_at(self._step, self._total, self.teacher_momentum)
        pairs: list[tuple[nn.Module, nn.Module]] = [
            (self.encoder, self.student),
            (self.teacher_head, self.student_head),
        ]
        for teacher, student in pairs:
            live = dict(student.named_parameters())
            for name, param in teacher.named_parameters():
                param.mul_(m).add_(live[name].detach(), alpha=1.0 - m)

    def embedding_surfaces(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        """The read backbone (the teacher's pooled embedding) and ``proj``, the teacher head's bottleneck on ``cls``."""
        backbone = self.encode(imgs)
        with torch.no_grad():
            tokens = self.encoder.forward(imgs.to(self.center.device))
            proj = self.teacher_head.bottleneck(tokens[:, 0])
        return {"backbone": backbone, "proj": proj}

    def make_views(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A positive pair: DINO's two global crops."""
        first, second = (t(imgs) for t in self.crops.globals)
        return first, second


def _strip(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """The entries under ``prefix`` (DDP's ``module.`` dropped first), prefix removed, ``patch_embed.proj`` remapped."""
    out = {}
    for raw, value in state.items():
        key = raw.removeprefix("module.")
        if key.startswith(prefix):
            out[key[len(prefix) :].replace("patch_embed.proj.", "patch_embed.")] = value
    return out


def _head_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A head's state with its last layer as ``weight_g`` / ``weight_v``.

    The released checkpoint stores the weight-normed layer as a plain ``last_layer.weight`` (rows of norm 1, the
    gain fixed at 1 by ``norm_last_layer``); it becomes ``weight_v`` with a unit gain, the same layer.
    """
    if "last_layer.weight" not in state:
        return state
    out = {k: v for k, v in state.items() if k != "last_layer.weight"}
    weight = state["last_layer.weight"]
    return {**out, "last_layer.weight_v": weight, "last_layer.weight_g": torch.ones(weight.shape[0], 1)}


def _load_backbone(backbone: nn.Module, state: dict[str, torch.Tensor], what: str) -> None:
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("_") for k in missing):  # only the input-normalization buffers may lack
        raise ValueError(f"DINO {what} load mismatch: missing={list(missing)} unexpected={list(unexpected)}")


def load_full_checkpoint(path: str | Path) -> dict[str, object]:
    """A DINO training checkpoint on CPU: ``student``, ``teacher`` and, when saved, ``args`` and ``dino_loss.center``.

    The released ViT-B/16 full checkpoint holds neither of the last two (only ``early_stopping_epoch``).
    """
    with torch.serialization.safe_globals([argparse.Namespace]):
        return dict(torch.load(Path(path), map_location="cpu", weights_only=True))


def build_dino(
    encoder: TinyViTEncoder,
    checkpoint: str | None = None,
    out_dim: int = 65536,
    hidden_dim: int = 2048,
    bottleneck_dim: int = 256,
    use_bn: bool = False,
    norm_last_layer: bool = True,
    global_size: int = 224,
    local_size: int = 96,
    n_local: int = 6,
    global_scale: tuple[float, float] = (0.4, 1.0),
    local_scale: tuple[float, float] = (0.05, 0.4),
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
    center_momentum: float = 0.9,
    teacher_momentum: float = 0.996,
    centering: bool = True,
    sharpening: bool = True,
) -> DINO:
    """Build :class:`DINO` around ``encoder`` (the teacher backbone), resumed from the full checkpoint if given.

    With ``checkpoint`` the student backbone and both heads come from it, and the centre and the head's shape too
    when it holds them (``dino_loss.center``; ``args``, overriding ``out_dim`` / ``use_bn`` / ``norm_last_layer``).
    A centre it lacks is estimated before step 0 (:meth:`DINO.prime`). The checkpoint's teacher backbone
    must equal ``encoder``'s weights bit for bit: the drive starts from the frozen floor itself. Without one
    (the synthetic tests) the student is a copy of the encoder and the heads are fresh.

    Raises:
        ValueError: If the checkpoint does not load cleanly, or its teacher differs from ``encoder``.
    """
    center = None
    student = copy.deepcopy(encoder)
    if checkpoint is not None:
        ckpt = load_full_checkpoint(checkpoint)
        args: argparse.Namespace | None = ckpt.get("args")  # type: ignore[assignment]
        if args is not None:
            out_dim = int(args.out_dim)
            use_bn = bool(args.use_bn_in_head)
            norm_last_layer = bool(args.norm_last_layer)
        teacher_state = _strip(ckpt["teacher"], "backbone.")  # type: ignore[arg-type]
        own = encoder.state_dict()
        differ = [k for k, v in teacher_state.items() if not torch.equal(own[k], v)]
        if differ:
            raise ValueError(f"the checkpoint's teacher differs from the configured encoder in {differ[:5]}.")
        _load_backbone(student, _strip(ckpt["student"], "backbone."), "student")  # type: ignore[arg-type]
        if "dino_loss" in ckpt:
            center = ckpt["dino_loss"]["center"]  # type: ignore[index]
        else:
            logger.info(f"{checkpoint} holds no centre: it is estimated before step 0")
    student_head, teacher_head = (
        DINOHead(
            int(encoder.embed_dim), out_dim, hidden_dim, bottleneck_dim, use_bn=use_bn, norm_last_layer=norm_last_layer
        )
        for _ in range(2)
    )
    if checkpoint is not None:
        student_head.load_state_dict(_head_state(_strip(ckpt["student"], "head.")))  # type: ignore[arg-type]
        teacher_head.load_state_dict(_head_state(_strip(ckpt["teacher"], "head.")))  # type: ignore[arg-type]
        logger.info(f"resumed DINO from {checkpoint} (student, teacher, heads; out_dim {out_dim})")
    else:
        teacher_head.load_state_dict(student_head.state_dict())
    return DINO(
        encoder,
        student,
        student_head,
        teacher_head,
        MultiCrop(
            global_size,
            local_size,
            n_local,
            (global_scale[0], global_scale[1]),  # a Hydra list arrives as a ListConfig
            (local_scale[0], local_scale[1]),
        ),
        student_temp=student_temp,
        teacher_temp=teacher_temp,
        center_momentum=center_momentum,
        teacher_momentum=teacher_momentum,
        centering=centering,
        sharpening=sharpening,
        center=center,
    )
