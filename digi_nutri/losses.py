"""Phase 1 losses. Geometry is affinity, not pixel feature L2."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from digi_nutri.protocol import VARIANTS, needs_aux, needs_geo, needs_sem


class LossWeights:
    def __init__(self, geo: float, sem: float, aux: float) -> None:
        self.geo = float(geo)
        self.sem = float(sem)
        self.aux = float(aux)


def task_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PMAE on original units plus Smooth L1 on train-split standardized units.

    Both terms average energy, mass, fat, carb, and protein together.
    """

    if pred.shape != target.shape or pred.shape[-1] != 5:
        raise ValueError(f"expected B,5 nutrients, got {tuple(pred.shape)}")
    pmae = (pred - target).abs() / target.abs().clamp_min(eps)
    pmae = pmae.mean()
    pred_n = (pred - mean) / std
    target_n = (target - mean) / std
    smooth = F.smooth_l1_loss(pred_n, target_n, beta=1.0)
    return pmae + smooth, {"pmae": pmae.detach(), "smooth_l1": smooth.detach()}


def spatial_affinity(feat: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity matrix over N = H*W regions. Shape B,N,N."""

    if feat.ndim != 4:
        raise ValueError("affinity expects B,C,H,W features")
    tokens = feat.flatten(2).transpose(1, 2)
    tokens = F.normalize(tokens, dim=-1, eps=1e-6)
    return torch.matmul(tokens, tokens.transpose(-1, -2))


def geo_affinity_loss(
    student_feats: list[torch.Tensor],
    teacher_feats: list[torch.Tensor],
) -> torch.Tensor:
    """Smooth-L1 between cosine affinity matrices. Not a pixel feature L2."""

    if len(student_feats) == 0 or len(student_feats) != len(teacher_feats):
        raise ValueError("student and teacher feature lists must match")
    total = student_feats[0].new_zeros(())
    for student, teacher in zip(student_feats, teacher_feats):
        teacher = teacher.detach()
        if student.shape[1] != teacher.shape[1]:
            raise ValueError(
                f"channel mismatch {student.shape} vs {teacher.shape}"
            )
        if student.shape[-2:] != teacher.shape[-2:]:
            student = F.adaptive_avg_pool2d(student, teacher.shape[-2:])
        total = total + F.smooth_l1_loss(
            spatial_affinity(student), spatial_affinity(teacher)
        )
    return total / len(student_feats)


def semantic_distill_loss(student_vec: torch.Tensor, teacher_vec: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 on L2-normalized ingredient-fused vectors."""

    student = F.normalize(student_vec.float(), dim=-1, eps=1e-6)
    teacher = F.normalize(teacher_vec.detach().float(), dim=-1, eps=1e-6)
    if student.shape != teacher.shape:
        raise ValueError(f"semantic shapes {student.shape} vs {teacher.shape}")
    return F.smooth_l1_loss(student, teacher)


def aux_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), target.float())


def phase1_total_loss(
    task: torch.Tensor,
    geo: torch.Tensor | None,
    sem: torch.Tensor | None,
    aux: torch.Tensor | None,
    scale: float,
    variant: str,
    weights: LossWeights,
) -> torch.Tensor:
    """B3 adds the unlocked terms. No per-sample teacher weight, no orthogonal term."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    if variant == "b1" or scale == 0.0:
        return task
    loss = task
    if needs_geo(variant):
        if geo is None:
            raise ValueError("L_geo is required")
        loss = loss + scale * weights.geo * geo
    if needs_sem(variant):
        if sem is None or aux is None:
            raise ValueError("L_sem and L_aux are required")
        loss = loss + scale * weights.sem * sem + scale * weights.aux * aux
    return loss
