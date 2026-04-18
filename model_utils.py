"""
model_utils.py
--------------
Pure, stateless functions for the PCA+GP surrogate model and evaluation metrics.

Architectural note: this is the abstraction boundary for the surrogate model.
To replace the PCA+GP proxy with a CPA-native implementation, swap out
`fit_gp_and_predict` with a function of the same signature that uses
CPA's perturbation embedding layer uncertainty (e.g. MC dropout or deep
ensembles). Everything in active_learner.py and generate_site_data.py
remains unchanged.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# PCA latent space construction
# ---------------------------------------------------------------------------

def build_pca_space(P_matrix: np.ndarray, n_components: int = 20, random_state: int = 42):
    """
    Fit PCA on the (n_perturbations × n_genes) profile matrix.

    Parameters
    ----------
    P_matrix      : (n_perts, n_genes) float array of per-perturbation mean profiles
    n_components  : number of PCA components to retain
    random_state  : random seed for reproducibility

    Returns
    -------
    pca           : fitted sklearn PCA object
    P_latent      : (n_perts, n_components) latent coordinates
    X_feat        : (n_perts, 10) first-10-PC features used as GP inputs
    cumvar        : cumulative variance explained by n_components PCs
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    P_latent = pca.fit_transform(P_matrix)
    X_feat = P_latent[:, :10].astype(np.float64)
    cumvar = float(pca.explained_variance_ratio_.sum())
    return pca, P_latent, X_feat, cumvar


# ---------------------------------------------------------------------------
# GP surrogate: fit on observed subset, predict for all perturbations
# ---------------------------------------------------------------------------

def fit_gp_and_predict(
    obs_idx: list,
    all_idx: list,
    X_feat: np.ndarray,
    P_latent: np.ndarray,
    pca,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a GP surrogate on observed perturbations and predict mean expression
    profiles + epistemic uncertainty for all perturbations.

    Strategy: one GP per PCA dimension (n_components GPs total).
    Predictions in PCA space are inverse-transformed back to gene space.
    Uncertainty = mean GP posterior std across all PCA dimensions.

    Parameters
    ----------
    obs_idx   : indices of currently observed perturbations
    all_idx   : indices of all perturbations (typically range(n_perts))
    X_feat    : (n_perts, 10) GP input features (first 10 PCs)
    P_latent  : (n_perts, n_components) full PCA coordinates (GP targets)
    pca       : fitted sklearn PCA object (for inverse_transform)

    Returns
    -------
    pred_mean : (n_perts, n_genes) predicted expression profiles
    pred_std  : (n_perts,) mean posterior std (epistemic uncertainty)
    """
    n_dims = P_latent.shape[1]
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

    X_obs = X_feat[obs_idx]
    Y_obs_pca = P_latent[obs_idx]  # (n_obs, n_dims)

    pred_pca = np.zeros((len(all_idx), n_dims))
    pred_std = np.zeros(len(all_idx))

    for dim in range(n_dims):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            normalize_y=True,
            random_state=42,
        )
        gp.fit(X_obs, Y_obs_pca[:, dim])
        mu, sigma = gp.predict(X_feat[all_idx], return_std=True)
        pred_pca[:, dim] = mu
        pred_std += sigma

    pred_std /= n_dims  # mean std across PCA dims → scalar per perturbation
    pred_mean = pca.inverse_transform(pred_pca)  # (n_perts, n_genes)
    return pred_mean, pred_std


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def top_deg_recovery(
    obs_idx: list,
    pred_mean: np.ndarray,
    Y_full: np.ndarray,
    ctrl_mean: np.ndarray,
    top_k: int = 20,
) -> float:
    """
    Mean fraction of true top-k DEGs correctly recovered in GP predictions,
    evaluated over all currently unobserved perturbations.

    A DEG is identified by |expression - control_mean|. Recovery = |true ∩ pred| / k.
    """
    n_total = Y_full.shape[0]
    unobs_idx = [i for i in range(n_total) if i not in set(obs_idx)]
    if not unobs_idx:
        return 1.0

    recoveries = []
    for i in unobs_idx:
        true_delta = np.abs(Y_full[i] - ctrl_mean)
        pred_delta = np.abs(pred_mean[i] - ctrl_mean)
        true_top = set(np.argsort(true_delta)[-top_k:])
        pred_top = set(np.argsort(pred_delta)[-top_k:])
        recoveries.append(len(true_top & pred_top) / top_k)

    return float(np.mean(recoveries))


def effect_size_spearman(
    obs_idx: list,
    pred_mean: np.ndarray,
    Y_full: np.ndarray,
    ctrl_mean: np.ndarray,
) -> float:
    """
    Spearman R between predicted and true perturbation effect sizes (L2 from
    control mean), evaluated over all currently unobserved perturbations.
    """
    n_total = Y_full.shape[0]
    unobs_idx = [i for i in range(n_total) if i not in set(obs_idx)]
    if not unobs_idx:
        return 1.0

    true_eff = [np.linalg.norm(Y_full[i] - ctrl_mean) for i in unobs_idx]
    pred_eff = [np.linalg.norm(pred_mean[i] - ctrl_mean) for i in unobs_idx]
    rho, _ = spearmanr(true_eff, pred_eff)
    return float(rho)


def compute_effect_sizes(Y_full: np.ndarray, ctrl_mean: np.ndarray) -> np.ndarray:
    """
    Compute L2 effect size for every perturbation relative to control mean.

    Returns
    -------
    effect_sizes : (n_perts,) float array
    """
    ctrl_f64 = ctrl_mean.astype(np.float64)
    return np.array([
        np.linalg.norm(Y_full[i].astype(np.float64) - ctrl_f64)
        for i in range(Y_full.shape[0])
    ])
