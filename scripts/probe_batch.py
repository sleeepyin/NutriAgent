"""Pick one batch size that fits T_geo, then use it for every Q0 model."""

from __future__ import annotations

import gc

import torch

from digi_nutri.models import select_backbone, build_model


def fits(backbone: str, batch: int) -> bool:
    torch.cuda.empty_cache()
    model = None
    optimizer = None
    model = build_model(
        "t_geo",
        backbone,
        pretrained=False,
        init_mass=200.0,
        init_density=[1.2, 0.06, 0.09, 0.08],
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    try:
        rgb = torch.randn(batch, 3, 384, 384, device="cuda")
        height = torch.randn(batch, 3, 384, 384, device="cuda")
        output = model(rgb, height)
        loss = output["yhat"].sum() + output["mass"].sum()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        return True
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return False
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the batch probe")
    backbone, count = select_backbone()
    print(f"backbone {backbone} params {count}", flush=True)
    chosen = None
    for batch in (32, 16, 8):
        ok = fits(backbone, batch)
        print(f"batch {batch} {'ok' if ok else 'oom'}", flush=True)
        if ok:
            chosen = batch
            break
    if chosen is None:
        raise SystemExit("T_geo does not fit at batch 8")
    print(f"BATCH {chosen}", flush=True)


if __name__ == "__main__":
    main()
