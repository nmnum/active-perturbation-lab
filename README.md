# Active Perturbation Lab

**Can adaptive selection reduce the number of CRISPR experiments needed to map a perturbation landscape?**

A proof-of-concept active learning loop on the Norman et al. 2019 K562 CRISPR screen (105 single-gene knockouts). A Gaussian process surrogate in PCA latent space selects which perturbations to observe next — revealing a systematic failure of standard UCB acquisition at high budgets.

Interactive demo: [autodiscovery-lab.lovable.app](https://autodiscovery-lab.lovable.app/)

---

## Key finding: UCB collapse

Standard UCB acquisition introduces systematic **selection bias** toward high-effect perturbations. By 30% budget all strategies achieve Spearman R ≈ 0.94 on unseen perturbations. But UCB exhausts the high-effect set, leaving low-effect perturbations unobserved — the GP extrapolates poorly to this regime, and Spearman R collapses to ~0.70 at 90% budget. Random selection, sampling uniformly, reaches ~0.99.

This identifies the **coverage–exploitation tradeoff** as the central design question for perturbation active learning — a gap not addressed by GeneDisco, NAIAD, or sequential OED.

---

## Repository structure

```
active_perturbation_lab/
├── model_utils.py            # PCA space, GP surrogate, evaluation metrics
├── active_learner.py         # Acquisition strategies + simulation loop
├── generate_site_data.py     # Runs simulation, exports active_learning_results.json
├── app.py                    # Streamlit interactive demo
├── requirements.txt
└── active_learning_results.json   # (generated — not tracked in git)
```

### Architectural note

`model_utils.py` is the surrogate model boundary. To replace the PCA+GP proxy with a CPA-native implementation, swap `fit_gp_and_predict` with a function of the same signature that uses CPA's perturbation embedding layer uncertainty (MC dropout or deep ensembles). Everything else is unchanged.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the data

You need the Norman et al. 2019 h5ad file from [scPerturb (Zenodo)](https://zenodo.org/records/7041849).

```bash
python generate_site_data.py \
    --data /path/to/NormanWeissman2019_filtered.h5ad \
    --out active_learning_results.json \
    --repeats 5
```

Expected runtime: ~5–15 minutes on a standard CPU. The script prints validation checks against the paper's reported numbers on completion.

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

The app runs in **demo mode** (synthetic data reproducing the paper's numbers) if `active_learning_results.json` is not found.

---

## Simulation parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Seed set | 10 perturbations | ~10% initial budget |
| Budget step | 5 per round | Perturbations added each round |
| Rounds | 17 | 10 + 17×5 = 95 perturbations → ~90% budget |
| Repeats | 5 | Independent random seed sets |
| β_UCB | 1.0 | Exploitation weight |
| β_div | 1.5 | Diversity weight (UCB + Diversity only) |
| PCA components | 20 | 85.6% variance explained |
| GP kernel | RBF + WhiteKernel | One GP per PCA dimension |

---

## Evaluation metrics

- **Spearman R**: Rank correlation between predicted and true perturbation effect sizes (L2 from control mean) across all currently *unobserved* perturbations — standard in CPA/GEARS benchmarks.
- **Top-20 DEG recovery**: Fraction of true top-20 differentially expressed genes per perturbation correctly identified in the GP's predicted top-20.

Both metrics are computed on the unobserved set only at each round.

---

## Next steps

1. **Model-native uncertainty** — replace PCA+GP with MC dropout or deep ensembles on CPA's perturbation embedding layer. Same acquisition function, calibrated uncertainty signal from the model that predicts.

2. **SP-FM uncertainty** — the Lotfollahi lab's SP-FM (ICLR 2026) conditions the base distribution on the perturbation; uncertainty over this base is a natural acquisition signal for active selection.

3. **Spatial perturbation selection** — extend to MintFlow: actively select which tissue niches to perturb, guided by NicheCompass niche uncertainty.

---

## Data

Norman et al. 2019 K562 CRISPRi screen, available via [scPerturb (Zenodo)](https://zenodo.org/records/7041849).

## Related work

- GeneDisco (Mehrjou et al. 2021)
- NAIAD (Qin et al. 2024)
- Sequential OED (Huang et al. 2023)
- CPA (Lotfollahi et al. 2023)
- GEARS (Roohani et al. 2023)
- SP-FM (Lotfollahi lab, ICLR 2026)
- MintFlow (Lotfollahi lab, 2025)

---

*Neha Mungale · MSc AI for Biomedicine and Healthcare, UCL*
