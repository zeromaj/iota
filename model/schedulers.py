import settings
from loguru import logger
import torch
import torch.optim as optim
import math


def make_lr_scheduler(optimizer: optim.lr_scheduler.LRScheduler | None = None):
    """
    Here are the stages of this scheduler:
    0. linear warm-up 0 → 1 × LRpeak
    1. constant plateau at LRpeak (optional)
    2. macro-cosine × micro-saw-tooth
    3. tail cosine to zero
    """

    # ─── hyper-parameters from settings.py ────────────────────────────
    warm_steps = settings.LR_WARMUP_STEPS  # e.g. 3_500
    plateau_steps = settings.LR_CONST_STEPS  # e.g.   500
    total_steps = settings.TOTAL_TRAIN_STEPS  # e.g. 100_000
    tail_frac = settings.LR_TAIL_STEPS_FRAC  # 0.02 (2 %)
    start_fac = settings.LR_WARMUP_START_FACTOR  # 0.002
    final_fac = settings.LR_FINAL_FACTOR  # 0.10
    cycle_length = settings.LR_SAW_CYCLE_LENGTH  # e.g. 10_000
    # if you prefer “N cycles”, set cycle_length = decay_steps // N

    tail_steps = int(total_steps * tail_frac)
    decay_steps = total_steps - warm_steps - plateau_steps - tail_steps
    assert decay_steps > 0, "decay phase would be zero/negative"

    # ─── phase-0: linear warm-up 0 → 1 × LRpeak ───────────────────────

    # If optimizer is None, create a mock opimizer as a single parameter
    if optimizer is None:
        optimizer = optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=settings.LEARNING_RATE)

    sched_warm = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_fac,
        end_factor=1.0,
        total_iters=warm_steps,
    )

    # ─── phase-1: constant plateau at LRpeak (optional) ───────────────
    sched_plateau = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0) if plateau_steps else None

    # ─── phase-2: macro-cosine × micro-saw-tooth  ────────────────────
    def combined_lambda(step):
        """
        step counts from 0 … decay_steps-1 inside the decay phase
        return LR multiplier ∈ [0, 1]
        """
        # ----- macro envelope  LRpeak → final_fac·LRpeak --------------
        macro_p = step / decay_steps
        macro = final_fac + (1.0 - final_fac) * 0.5 * (1 + math.cos(math.pi * macro_p))

        # ----- micro cosine-restart 1 → 0.1 → 1 every cycle_length ----
        cycle_p = (step % cycle_length) / cycle_length
        micro = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * cycle_p))
        # micro ∈ [0.1, 1]

        return macro * micro  # overall multiplier

    sched_saw = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=combined_lambda)

    # ─── phase-3: tail cosine to zero ─────────────────────────────────
    def tail_lambda(step):
        p = step / tail_steps
        return final_fac * 0.5 * (1 + math.cos(math.pi * p))  # ↘ 0

    sched_tail = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=tail_lambda)

    # ─── stitch phases together ──────────────────────────────────────
    schedulers = [sched_warm]
    milestones = [warm_steps]

    if sched_plateau:
        schedulers.append(sched_plateau)
        milestones.append(milestones[-1] + plateau_steps)

    schedulers += [sched_saw, sched_tail]
    milestones += [milestones[-1] + decay_steps]  # (= total)

    lr_scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

    logger.info(
        f"LR schedule\n"
        f"  warm-up   : 0–{warm_steps-1}\n"
        f"  plateau   : {warm_steps}–{warm_steps+plateau_steps-1}\n"
        f"  saw-tooth : {milestones[-2]-decay_steps}–{milestones[-2]-1} "
        f"(cycle_length={cycle_length})\n"
        f"  tail      : {milestones[-2]}–{total_steps-1}"
    )

    return lr_scheduler
