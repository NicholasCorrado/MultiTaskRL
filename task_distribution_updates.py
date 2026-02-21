from typing import Optional
import numpy as np


def kl_project_with_floor(z: np.ndarray, eps: float) -> np.ndarray:
    """
    KL projection of distribution z onto:
        { q : q_i >= eps, sum_i q_i = 1 }
    Solves:  min_q KL(q || z)  subject to q_i >= dro_eps.
    """
    k = len(z)

    free = np.ones(k, dtype=bool)
    q = np.zeros_like(z)

    while True:
        mass_free = 1.0 - eps * (~free).sum()
        if mass_free < 0:
            return np.ones(k) / k

        z_free_sum = z[free].sum()
        if z_free_sum == 0:
            q[free] = mass_free / free.sum()
        else:
            q[free] = (mass_free / z_free_sum) * z[free]

        q[~free] = eps

        violated = free & (q < eps - 1e-12)
        if not violated.any():
            break
        free[violated] = False

    return q / q.sum()


def mirror_ascent_kl_update(
        q: np.ndarray,
        gap: np.ndarray,
        eta: float,
        step_size: float,
        eps: Optional[float] = None,
        p0: Optional[np.ndarray] = None,
) -> np.ndarray:
    gap = np.asarray(gap, dtype=np.float64)
    k = len(q)

    if p0 is None:
        p0 = np.ones(k, dtype=np.float64) / k
    else:
        p0 = np.asarray(p0, dtype=np.float64)
        p0 = p0 / p0.sum()

    alpha = step_size  # now step_size is directly in [0, 1]

    log_q_star = np.log(p0) + eta * gap
    log_q_new = alpha * log_q_star + (1.0 - alpha) * np.log(q)

    log_q_new -= np.max(log_q_new)
    q_new = np.exp(log_q_new)
    q_new /= q_new.sum()

    if eps is not None and eps > 0:
        q_new = kl_project_with_floor(q_new, eps)

    return q_new


def learning_progress_update(
        q: np.ndarray,
        learning_progress: np.ndarray,
        success_rates: np.ndarray,
        eta: float,
        step_size: float,
        eps: Optional[float] = None,
        p0: Optional[np.ndarray] = None,
        success_threshold: float = 0.9,
) -> np.ndarray:
    q_new = mirror_ascent_kl_update(
        q=q,
        gap=learning_progress,
        eta=eta,
        step_size=step_size,
        eps=None, # no need to do kl projection here since we do it later
        p0=p0,  # uniform by default
    )

    solved = np.where(success_rates > success_threshold)[0]
    q_new[solved] = eps
    q_new = kl_project_with_floor(q_new, eps=eps)

    return q_new

def easy_first_curriculum_update(
    success_rates: np.ndarray,
    eps: float,
    success_threshold: float = 0.9,
) -> np.ndarray:
    k = len(success_rates)

    # leftmost unsolved task
    unsolved = np.where(success_rates < success_threshold)[0]
    if len(unsolved) == 0:
        q = np.ones(k) / k
        return q / q.sum()

    active = int(unsolved[0])

    q = np.full(k, eps)
    q[active] = 1.0 - eps * (k - 1)
    return q / q.sum()

def hard_first_curriculum_update(
        success_rates: np.ndarray,
        eps: float,
        success_threshold: float = 0.9,
) -> np.ndarray:
    k = len(success_rates)

    # rightmost unsolved task
    unsolved = np.where(success_rates < success_threshold)[0]
    if len(unsolved) == 0:
        q = np.ones(k) / k
        return q / q.sum()

    active = int(unsolved[-1])

    q = np.full(k, eps)
    q[active] = 1.0 - eps * (k - 1)
    return q / q.sum()

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