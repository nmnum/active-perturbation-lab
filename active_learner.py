"""
active_learner.py
-----------------
Active learning simulation loop and acquisition strategy implementations.

Three strategies:
  - random      : uniform random baseline
  - ucb_only    : UCB (predicted effect size + GP uncertainty)
  - ucb_diverse : UCB + diversity penalty (distance to observed set in PCA space)

The simulation loop returns full selection sequences (not just metrics), so
generate_site_data.py can export exactly which perturbations were selected
at every round for every strategy and repeat — powering the interactive demo.
"""

import numpy as np
from model_utils import fit_gp_and_predict, top_deg_recovery, effect_size_spearman


# ---------------------------------------------------------------------------
# Acquisition scores
# ---------------------------------------------------------------------------

def _score_random(unobserved: list, rng: np.random.RandomState, budget_step: int) -> list:
    """Uniform random selection — baseline."""
    chosen = list(rng.choice(unobserved, size=min(budget_step, len(unobserved)), replace=False))
    return chosen


def _score_ucb_only(
    unobserved: list,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    ctrl_mean: np.ndarray,
    budget_step: int,
    beta_ucb: float = 1.0,
) -> list:
    """
    UCB acquisition: normalised predicted effect size + beta_ucb * normalised uncertainty.
    Selects top-k unobserved perturbations by score.
    """
    ctrl_f64 = ctrl_mean.astype(np.float64)
    eff_vals = np.array([np.linalg.norm(pred_mean[i].astype(np.float64) - ctrl_f64)
                         for i in unobserved])
    unc_vals = pred_std[unobserved]

    eff_norm = (eff_vals - eff_vals.min()) / (np.ptp(eff_vals) + 1e-10)
    unc_norm = (unc_vals - unc_vals.min()) / (np.ptp(unc_vals) + 1e-10)
    score = eff_norm + beta_ucb * unc_norm

    order = np.argsort(score)[::-1]
    return [unobserved[j] for j in order[:budget_step]]


def _score_ucb_diverse(
    unobserved: list,
    observed: list,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    X_feat: np.ndarray,
    ctrl_mean: np.ndarray,
    budget_step: int,
    beta_ucb: float = 1.0,
    beta_div: float = 1.5,
) -> list:
    """
    UCB + Diversity acquisition: UCB score augmented with minimum distance to
    the observed set in PCA latent space. Promotes coverage of the perturbation
    landscape, mitigating the selection bias of UCB-only.
    """
    ctrl_f64 = ctrl_mean.astype(np.float64)
    eff_vals = np.array([np.linalg.norm(pred_mean[i].astype(np.float64) - ctrl_f64)
                         for i in unobserved])
    unc_vals = pred_std[unobserved]
    X_obs_feat = X_feat[observed]
    div_vals = np.array([
        np.min(np.linalg.norm(X_feat[i] - X_obs_feat, axis=1))
        for i in unobserved
    ])

    eff_norm = (eff_vals - eff_vals.min()) / (np.ptp(eff_vals) + 1e-10)
    unc_norm = (unc_vals - unc_vals.min()) / (np.ptp(unc_vals) + 1e-10)
    div_norm = (div_vals - div_vals.min()) / (np.ptp(div_vals) + 1e-10)
    score = eff_norm + beta_ucb * unc_norm + beta_div * div_norm

    order = np.argsort(score)[::-1]
    return [unobserved[j] for j in order[:budget_step]]


# ---------------------------------------------------------------------------
# Single-repeat simulation loop
# ---------------------------------------------------------------------------

def run_active_learning(
    strategy: str,
    seed_idx: list,
    all_idx: list,
    X_feat: np.ndarray,
    P_latent: np.ndarray,
    Y_full: np.ndarray,
    ctrl_mean: np.ndarray,
    pca,
    n_rounds: int,
    budget_step: int,
    rng: np.random.RandomState,
    beta_ucb: float = 1.0,
    beta_div: float = 1.5,
) -> dict:
    """
    Run one repeat of the active learning simulation for a given strategy.

    At each round:
      1. Fit GP surrogate on currently observed perturbations
      2. Record metrics (Spearman R, DEG recovery) on unobserved set
      3. Select next `budget_step` perturbations using the acquisition function
      4. Record the full cumulative observed set (for visualisation replay)

    Parameters
    ----------
    strategy    : one of 'random', 'ucb_only', 'ucb_diverse'
    seed_idx    : initial seed perturbation indices
    all_idx     : all perturbation indices
    X_feat      : (n_perts, 10) GP input features
    P_latent    : (n_perts, n_components) PCA latent coordinates
    Y_full      : (n_perts, n_genes) ground-truth expression profiles
    ctrl_mean   : (n_genes,) control mean expression profile
    pca         : fitted sklearn PCA object
    n_rounds    : number of active learning rounds
    budget_step : perturbations added per round
    rng         : seeded numpy RandomState (for random strategy)
    beta_ucb    : UCB exploitation weight
    beta_div    : diversity weight (ucb_diverse only)

    Returns
    -------
    dict with keys:
      'seed_indices'  : list — initial seed set
      'rounds'        : list of dicts, one per round:
                          budget_pct      : float (0–100)
                          observed_indices: list (cumulative, for scatter replay)
                          spearman_r      : float
                          deg_recovery    : float
    """
    observed = list(seed_idx)
    rounds = []

    for round_i in range(n_rounds + 1):
        budget_pct = round(len(observed) / len(all_idx) * 100, 2)
        unobserved = [i for i in all_idx if i not in set(observed)]

        pred_mean, pred_std = fit_gp_and_predict(
            observed, all_idx, X_feat, P_latent, pca
        )

        deg_rec = top_deg_recovery(observed, pred_mean, Y_full, ctrl_mean)
        spear = effect_size_spearman(observed, pred_mean, Y_full, ctrl_mean)

        rounds.append({
            "budget_pct": budget_pct,
            "observed_indices": list(observed),  # cumulative — drives scatter replay
            "spearman_r": round(float(spear), 5),
            "deg_recovery": round(float(deg_rec), 5),
        })

        # Stop after recording final-round metrics
        if round_i == n_rounds or not unobserved:
            break

        # Acquisition
        if strategy == "random":
            chosen = _score_random(unobserved, rng, budget_step)
        elif strategy == "ucb_only":
            chosen = _score_ucb_only(
                unobserved, pred_mean, pred_std, ctrl_mean, budget_step, beta_ucb
            )
        elif strategy == "ucb_diverse":
            chosen = _score_ucb_diverse(
                unobserved, observed, pred_mean, pred_std,
                X_feat, ctrl_mean, budget_step, beta_ucb, beta_div
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        observed.extend(chosen)

    return {
        "seed_indices": list(seed_idx),
        "rounds": rounds,
    }


# ---------------------------------------------------------------------------
# Full simulation: all strategies × all repeats
# ---------------------------------------------------------------------------

def run_all_strategies(
    all_idx: list,
    X_feat: np.ndarray,
    P_latent: np.ndarray,
    Y_full: np.ndarray,
    ctrl_mean: np.ndarray,
    pca,
    n_repeats: int = 5,
    seed_size: int = 10,
    n_rounds: int = 17,
    budget_step: int = 5,
    beta_ucb: float = 1.0,
    beta_div: float = 1.5,
    verbose: bool = True,
) -> dict:
    """
    Run the full simulation: 3 strategies × n_repeats × n_rounds.

    Seed construction exactly matches the original notebook:
        rng = np.random.RandomState(repeat * 13 + 7)
    All three strategies receive the same seed set per repeat, with
    independent RNG state for the random strategy.

    Returns
    -------
    dict: { 'random': [repeat_0_result, ...], 'ucb_only': [...], 'ucb_diverse': [...] }
    Each repeat_result is the dict returned by run_active_learning().
    """
    strategies = ["random", "ucb_only", "ucb_diverse"]
    results = {s: [] for s in strategies}

    for repeat in range(n_repeats):
        rng = np.random.RandomState(repeat * 13 + 7)
        seed_idx = [int(i) for i in rng.choice(all_idx, size=seed_size, replace=False)]

        for strategy in strategies:
            # Reset rng to same state: same seed set, independent runs per strategy
            rng_s = np.random.RandomState(repeat * 13 + 7)
            result = run_active_learning(
                strategy=strategy,
                seed_idx=seed_idx,
                all_idx=all_idx,
                X_feat=X_feat,
                P_latent=P_latent,
                Y_full=Y_full,
                ctrl_mean=ctrl_mean,
                pca=pca,
                n_rounds=n_rounds,
                budget_step=budget_step,
                rng=rng_s,
                beta_ucb=beta_ucb,
                beta_div=beta_div,
            )
            results[strategy].append(result)

        if verbose:
            seed_r = results["random"][-1]["rounds"][0]["spearman_r"]
            seed_u = results["ucb_only"][-1]["rounds"][0]["spearman_r"]
            seed_d = results["ucb_diverse"][-1]["rounds"][0]["spearman_r"]
            fin_r = results["random"][-1]["rounds"][-1]["spearman_r"]
            fin_u = results["ucb_only"][-1]["rounds"][-1]["spearman_r"]
            fin_d = results["ucb_diverse"][-1]["rounds"][-1]["spearman_r"]
            print(f"Repeat {repeat + 1}/{n_repeats}:")
            print(f"  Seed  Spearman R: random={seed_r:.3f}  ucb={seed_u:.3f}  diverse={seed_d:.3f}")
            print(f"  Final Spearman R: random={fin_r:.3f}  ucb={fin_u:.3f}  diverse={fin_d:.3f}")

    return results


# ---------------------------------------------------------------------------
# Selection bias helper: re-run to 50% budget for Panel C
# ---------------------------------------------------------------------------

def selection_bias_at_50pct(
    all_idx: list,
    X_feat: np.ndarray,
    P_latent: np.ndarray,
    Y_full: np.ndarray,
    ctrl_mean: np.ndarray,
    effect_sizes: np.ndarray,
    pca,
    budget_step: int = 5,
    seed_size: int = 10,
) -> dict:
    """
    Re-run repeat 0 (seed=RandomState(7)) up to 50% budget for each strategy.
    Returns selected indices and their true effect sizes — powers Panel C.

    Returns
    -------
    dict: { 'random': {'selected_indices': [...], 'effect_sizes': [...]}, ... }
    """
    n_total = len(all_idx)
    n_at_50 = int(0.5 * n_total)
    n_rounds_50 = int(np.ceil((n_at_50 - seed_size) / budget_step))

    strategies = ["random", "ucb_only", "ucb_diverse"]
    bias_results = {}

    for strategy in strategies:
        rng0 = np.random.RandomState(7)
        seed_idx0 = list(rng0.choice(all_idx, size=seed_size, replace=False))
        observed = list(seed_idx0)
        rng_s = np.random.RandomState(7)

        for _ in range(n_rounds_50):
            unobserved = [i for i in all_idx if i not in set(observed)]
            if not unobserved or len(observed) >= n_at_50:
                break

            pred_mean, pred_std = fit_gp_and_predict(
                observed, all_idx, X_feat, P_latent, pca
            )

            if strategy == "random":
                chosen = _score_random(unobserved, rng_s, budget_step)
            elif strategy == "ucb_only":
                chosen = _score_ucb_only(unobserved, pred_mean, pred_std, ctrl_mean, budget_step)
            elif strategy == "ucb_diverse":
                chosen = _score_ucb_diverse(
                    unobserved, observed, pred_mean, pred_std,
                    X_feat, ctrl_mean, budget_step
                )

            observed.extend(chosen)

        selected = observed[:n_at_50]
        bias_results[strategy] = {
            "selected_indices": selected,
            "effect_sizes": [round(float(effect_sizes[i]), 5) for i in selected],
        }

    return bias_results
