from typing import Optional
import numpy as np


def kl_project_with_floor(z: np.ndarray, dro_eps: float) -> np.ndarray:
    """
    KL projection of distribution z onto:
        { q : q_i >= dro_eps, sum_i q_i = 1 }
    Solves:  min_q KL(q || z)  subject to q_i >= dro_eps.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()
    k = len(z)

    free = np.ones(k, dtype=bool)
    q = np.zeros_like(z)

    while True:
        num_clipped = (~free).sum()
        mass_free = 1.0 - dro_eps * num_clipped
        if mass_free < 0:
            # infeasible floor; fallback to uniform
            return np.ones(k, dtype=np.float64) / k

        z_free_sum = z[free].sum()
        if z_free_sum == 0:
            q[free] = mass_free / free.sum()
        else:
            scale = mass_free / z_free_sum
            q[free] = scale * z[free]

        q[~free] = dro_eps

        violated = free & (q < dro_eps - 1e-12)
        if not violated.any():
            break
        free[violated] = False

    q /= q.sum()
    return q


def kl_regularized_dro_update(
        q: np.ndarray,
        gap: np.ndarray,
        eta: float,
        step_size: float,
        p0: Optional[np.ndarray] = None,
        dro_eps: Optional[float] = None,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    gap = np.asarray(gap, dtype=np.float64)
    k = len(q)

    if p0 is None:
        p0 = np.ones(k, dtype=np.float64) / k
    else:
        p0 = np.asarray(p0, dtype=np.float64)
        p0 = p0 / p0.sum()

    alpha = step_size  # now step_size is directly in [0, 1]

    q_safe = np.clip(q, 1e-30, 1.0)
    p0_safe = np.clip(p0, 1e-30, 1.0)

    log_q_star = np.log(p0_safe) + eta * gap
    log_q_new = alpha * log_q_star + (1.0 - alpha) * np.log(q_safe)

    log_q_new -= np.max(log_q_new)
    q_new = np.exp(log_q_new)
    q_new /= q_new.sum()

    if dro_eps is not None and dro_eps > 0:
        q_new = kl_project_with_floor(q_new, dro_eps)

    # print(q_new)
    return q_new

def curriculum_update(
        q: np.ndarray,
        gap: np.ndarray,
        eta: float,          # unused (kept for drop-in compatibility)
        step_size: float,    # unused (kept for drop-in compatibility)
        p0: Optional[np.ndarray] = None,   # unused (kept for drop-in compatibility)
        dro_eps: Optional[float] = None,   # treat dro_eps as eps
        threshold: float = 0.1,
) -> np.ndarray:
    """
    Curriculum sampler:
      - Maintain an 'active' task that starts at index 0.
      - Put probability mass 1 - eps*(k-1) on the active task and eps on all others.
      - When gap[active] >= threshold, advance active to active+1 (until k-1).
    Notes:
      - We infer the current active task from q (argmax).
      - dro_eps is treated as eps.
    """
    q = np.asarray(q, dtype=np.float64)
    gap = np.asarray(gap, dtype=np.float64)
    k = len(q)

    eps = 0.0 if dro_eps is None else float(dro_eps)

    # infer current active task (start at 0 if q is degenerate/invalid)
    if q.size == 0 or not np.isfinite(q).all() or q.sum() <= 0:
        active = 0
    else:
        active = int(np.argmax(q))

    # advance if solved (until last task)
    if 0 <= active < k - 1 and gap[active] <= threshold:
        active += 1

    # construct curriculum distribution
    q_new = np.full(k, eps, dtype=np.float64)

    # ensure nonnegative mass on active
    active_mass = 1.0 - eps * (k - 1)
    if active_mass < 0.0:
        raise ValueError(f"eps={eps} too large for k={k}: need eps <= 1/(k-1).")

    q_new[active] = active_mass

    if np.all(gap[active] <= threshold):
        q_new = np.ones(k) * 1/k

    if dro_eps is not None and dro_eps > 0:
        q_new = kl_project_with_floor(q_new, dro_eps)
    return q_new


def smt_update(
        q: np.ndarray,
        min_threshold: np.ndarray,
        max_threshold: np.ndarray,
        score: np.ndarray,
        eta: float,
        step_size: float,
        p0: Optional[np.ndarray] = None,
        dro_eps: Optional[float] = None,
) -> np.ndarray:
    k = len(q)

    if p0 is None:
        p0 = np.ones(k, dtype=np.float64) / k
    else:
        p0 = np.asarray(p0, dtype=np.float64)
        p0 = p0 / p0.sum()

    if min_threshold is None:
        min_threshold = np.ones(k) * 0.1
    if max_threshold is None:
        max_threshold = np.ones(k) * 0.9

    alpha = step_size  # now step_size is directly in [0, 1]

    q_safe = np.clip(q, 1e-30, 1.0)
    p0_safe = np.clip(p0, 1e-30, 1.0)

    tasks_not_learned = score <= min_threshold
    tasks_learned = score >= max_threshold
    tasks_learning = ~(tasks_not_learned | tasks_learned)

    log_q_star = np.log(p0_safe) + eta * gap
    log_q_new = alpha * log_q_star + (1.0 - alpha) * np.log(q_safe)

    log_q_new -= np.max(log_q_new)
    q_new = np.exp(log_q_new)
    q_new /= q_new.sum()

    if dro_eps is not None and dro_eps > 0:
        q_new = kl_project_with_floor(q_new, dro_eps)

    # print(q_new)
    return q_new

def exponentiated_gradient_ascent_step(
    args,
    w: np.ndarray,
    returns: np.ndarray,
    returns_ref: np.ndarray,
    previous_return_avg: np.ndarray,
    learning_rate: float = 1.0,
    eps: float = 0.1,
) -> np.ndarray:

    diff = np.clip(returns_ref - returns, 0, np.inf)
    w_new = w * np.exp(learning_rate * diff)
    w_new = w_new / w_new.sum()

    w_uniform = np.ones_like(w_new) / len(w_new)
    w_new = (1 - eps) * w_new + eps * w_uniform
    return w_new