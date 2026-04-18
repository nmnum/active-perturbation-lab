"""
generate_site_data.py
---------------------
Orchestration script: loads the Norman et al. 2019 h5ad file, runs the full
active learning simulation, and exports active_learning_results.json.

This file contains NO ML logic — it only orchestrates model_utils.py and
active_learner.py and serialises their output to a structured JSON payload
that powers the Streamlit demo without any live GP computation.

Usage
-----
    python generate_site_data.py [--data PATH] [--out PATH] [--repeats N]

Defaults:
    --data     norman_data/NormanWeissman2019_filtered.h5ad
    --out      active_learning_results.json
    --repeats  5
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

warnings.filterwarnings("ignore")

from model_utils import build_pca_space, compute_effect_sizes
from active_learner import run_all_strategies, selection_bias_at_50pct

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle NumPy types during JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate active_learning_results.json")
    p.add_argument("--data", default="./NormanWeissman2019_filtered.h5ad",
                   help="Path to NormanWeissman2019_filtered.h5ad")
    p.add_argument("--out", default="active_learning_results.json",
                   help="Output JSON path")
    p.add_argument("--repeats", type=int, default=5,
                   help="Number of independent simulation repeats")
    p.add_argument("--seed-size", type=int, default=10)
    p.add_argument("--budget-step", type=int, default=5)
    p.add_argument("--n-rounds", type=int, default=17)
    p.add_argument("--beta-ucb", type=float, default=1.0)
    p.add_argument("--beta-div", type=float, default=1.5)
    p.add_argument("--n-hvgs", type=int, default=2000)
    p.add_argument("--pca-components", type=int, default=20)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(h5ad_path: str, n_hvgs: int = 2000):
    """
    Load Norman et al. 2019, subset to single-gene perturbations + controls,
    normalise, log1p, select HVGs. Returns preprocessed AnnData.
    """
    print(f"Loading {h5ad_path} ...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Raw: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Subset to single-gene perturbations + controls
    adata_sub = adata[
        (adata.obs["nperts"] == 1) | (adata.obs["perturbation"] == "control")
    ].copy()

    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)
    sc.pp.highly_variable_genes(adata_sub, n_top_genes=n_hvgs, flavor="seurat")
    adata_hvg = adata_sub[:, adata_sub.var["highly_variable"]].copy()

    print(f"  After HVG selection: {adata_hvg.shape}")
    return adata_hvg


def build_ground_truth(adata_hvg):
    """
    Compute per-perturbation mean expression profiles and control mean.
    Returns perturbation names list, profile matrix, ctrl_mean.
    """
    ctrl_cells = adata_hvg[adata_hvg.obs["perturbation"] == "control"]
    ctrl_mean = np.asarray(ctrl_cells.X.mean(axis=0)).flatten()

    perturbations = sorted([
        p for p in adata_hvg.obs["perturbation"].unique() if p != "control"
    ])

    profiles = {}
    for pert in perturbations:
        cells = adata_hvg[adata_hvg.obs["perturbation"] == pert]
        profiles[pert] = np.asarray(cells.X.mean(axis=0)).flatten()

    P_matrix = np.vstack([profiles[p] for p in perturbations]).astype(np.float32)

    print(f"  Perturbations: {len(perturbations)}")
    print(f"  Profile matrix: {P_matrix.shape}")
    print(f"  Control cells: {ctrl_cells.n_obs}")
    return perturbations, P_matrix, ctrl_mean


# ---------------------------------------------------------------------------
# JSON assembly
# ---------------------------------------------------------------------------

def build_json_payload(
    perturbations: list,
    P_latent: np.ndarray,
    effect_sizes: np.ndarray,
    results: dict,
    bias_results: dict,
    metadata: dict,
) -> dict:
    """
    Assemble the full JSON payload from simulation results.

    Schema
    ------
    {
      "metadata": { n_perturbations, seed_size, budget_step, ... },
      "perturbations": [ { id, pca_x, pca_y, effect_size }, ... ],
      "strategies": {
        "random": { "repeats": [ { seed_indices, rounds: [ { budget_pct,
                                    observed_indices, spearman_r,
                                    deg_recovery } ] } ] },
        "ucb_only": ...,
        "ucb_diverse": ...
      },
      "selection_bias_rep0": {
        "random":      { selected_indices, effect_sizes },
        "ucb_only":    { selected_indices, effect_sizes },
        "ucb_diverse": { selected_indices, effect_sizes }
      }
    }

    pca_x / pca_y are PC1 and PC2 (used for the 2D scatter visualisation).
    """
    # Perturbation landscape (static data)
    perts_payload = [
        {
            "id": perturbations[i],
            "pca_x": round(float(P_latent[i, 0]), 5),
            "pca_y": round(float(P_latent[i, 1]), 5),
            "effect_size": round(float(effect_sizes[i]), 5),
        }
        for i in range(len(perturbations))
    ]

    # Strategy results (dynamic data)
    strategies_payload = {}
    for strategy, repeats in results.items():
        strategies_payload[strategy] = {"repeats": repeats}

    return {
        "metadata": metadata,
        "perturbations": perts_payload,
        "strategies": strategies_payload,
        "selection_bias_rep0": bias_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Active Perturbation Lab — generate_site_data.py")
    print("=" * 60)

    # 1. Load and preprocess
    print("\n[1/5] Loading and preprocessing data ...")
    adata_hvg = load_and_preprocess(args.data, n_hvgs=args.n_hvgs)

    # 2. Build ground truth
    print("\n[2/5] Building ground-truth profiles ...")
    perturbations, P_matrix, ctrl_mean = build_ground_truth(adata_hvg)
    ctrl_mean_f64 = ctrl_mean.astype(np.float64)
    Y_full = P_matrix.astype(np.float64)
    all_idx = list(range(len(perturbations)))

    # 3. PCA latent space
    print(f"\n[3/5] Fitting PCA ({args.pca_components} components) ...")
    pca, P_latent, X_feat, cumvar = build_pca_space(
        P_matrix, n_components=args.pca_components
    )
    print(f"  Cumulative variance explained: {cumvar:.1%}")

    effect_sizes = compute_effect_sizes(Y_full, ctrl_mean_f64)
    print(f"  Effect size range: {effect_sizes.min():.3f} – {effect_sizes.max():.3f}")

    # 4. Run full simulation
    print(f"\n[4/5] Running simulation ({args.repeats} repeats × 3 strategies × {args.n_rounds} rounds) ...")
    print("  This takes ~5–15 minutes on a standard CPU. Grab a coffee.\n")

    results = run_all_strategies(
        all_idx=all_idx,
        X_feat=X_feat,
        P_latent=P_latent,
        Y_full=Y_full,
        ctrl_mean=ctrl_mean_f64,
        pca=pca,
        n_repeats=args.repeats,
        seed_size=args.seed_size,
        n_rounds=args.n_rounds,
        budget_step=args.budget_step,
        beta_ucb=args.beta_ucb,
        beta_div=args.beta_div,
        verbose=True,
    )

    # 5. Selection bias (Panel C data)
    print("\n[5/5] Computing selection-bias distributions (repeat 0, 50% budget) ...")
    bias_results = selection_bias_at_50pct(
        all_idx=all_idx,
        X_feat=X_feat,
        P_latent=P_latent,
        Y_full=Y_full,
        ctrl_mean=ctrl_mean_f64,
        effect_sizes=effect_sizes,
        pca=pca,
        budget_step=args.budget_step,
        seed_size=args.seed_size,
    )
    for strat, b in bias_results.items():
        mean_eff = np.mean(b["effect_sizes"])
        print(f"  {strat:15s}: mean effect size at 50% budget = {mean_eff:.3f}")
    print("  Expected: random≈2.33, ucb_only≈3.11, ucb_diverse≈3.10")

    # Assemble and write JSON
    metadata = {
        "n_perturbations": len(perturbations),
        "n_genes_hvg": int(Y_full.shape[1]),
        "pca_components": args.pca_components,
        "pca_variance_explained": round(cumvar, 4),
        "seed_size": args.seed_size,
        "budget_step": args.budget_step,
        "n_rounds": args.n_rounds,
        "n_repeats": args.repeats,
        "beta_ucb": args.beta_ucb,
        "beta_div": args.beta_div,
        "dataset": "Norman et al. 2019 (K562 CRISPRi, scPerturb)",
        "generated_by": "generate_site_data.py",
    }

    payload = build_json_payload(
        perturbations=perturbations,
        P_latent=P_latent,
        effect_sizes=effect_sizes,
        results=results,
        bias_results=bias_results,
        metadata=metadata,
    )

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)

    size_mb = out_path.stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"\nDone. JSON written to: {out_path}  ({size_mb:.1f} MB)")
    print(f"Total time: {elapsed / 60:.1f} min")
    print("\nVerification checks:")

    # Sanity checks
    n_perts = len(payload["perturbations"])
    n_rep_random = len(payload["strategies"]["random"]["repeats"])
    n_rounds_rep0 = len(payload["strategies"]["random"]["repeats"][0]["rounds"])
    print(f"  Perturbations in JSON   : {n_perts}  (expected 105)")
    print(f"  Repeats per strategy    : {n_rep_random}  (expected {args.repeats})")
    print(f"  Rounds in repeat 0      : {n_rounds_rep0}  (expected {args.n_rounds + 1})")

    # Reproduce key paper numbers
    def _spearman_at_budget(strategy, target_pct):
        rep_results = payload["strategies"][strategy]["repeats"]
        vals = []
        for rep in rep_results:
            rounds = rep["rounds"]
            closest = min(rounds, key=lambda r: abs(r["budget_pct"] - target_pct))
            vals.append(closest["spearman_r"])
        return np.mean(vals), np.std(vals)

    mu30, sd30 = _spearman_at_budget("random", 30)
    mu90u, sd90u = _spearman_at_budget("ucb_only", 90)
    mu90r, sd90r = _spearman_at_budget("random", 90)
    print(f"\n  Spearman R @ 30% budget (random)     : {mu30:.3f} ± {sd30:.3f}  (expect ~0.94)")
    print(f"  Spearman R @ 90% budget (ucb_only)   : {mu90u:.3f} ± {sd90u:.3f}  (expect ~0.70)")
    print(f"  Spearman R @ 90% budget (random)     : {mu90r:.3f} ± {sd90r:.3f}  (expect ~0.99)")


if __name__ == "__main__":
    main()
