# Project Title

This project test the performance of doubly robust estimators for finding the optimal dynamic treatment regimes where the propensity and outcome models are non-parametric.

---

## Overview

This repository contains code for simulating four main data sets and finding their optimal treatment regime via parametric and non-parametric DREs.

The project includes:
- Data generation and simulation code
- Statistical models (parametric and machine learning)
- Evaluation of estimators via repeated sampling
- Visualization and summary of results

The primary focus is on **causal inference and dynamic treatment regimes**.

---

## Repository Structure

```text
.
├── README.md                 # Project documentation
├── R scripts/                # R scripts for data generation, models, analysis
│   ├── run_sims.R
│   ├── run_one_sim.R
│   ├── utils.R
│   └── plots.R
├── python/                   # Python scripts (if applicable)
│   ├── run_sims.py
│   ├── run_one_sim.py
│   ├── utils.py
│   └── plots.py
├── output/                   # Generated figures, tables, results
└── figures/                  # Saved plots