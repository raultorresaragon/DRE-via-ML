# Doubly Robust Estimation via ML for Optimal Treatment Regimes

This project tests the performance of doubly robust estimators (DRE) for finding the optimal dynamic treatment regimes (DTR) where:  
- the propensity and outcome models are non-parametric, 
- the conditional expectation function of the outcome given the treatment and covariates is non-linear,
- and the treatment variable has more than two levels. 

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
├── R_scripts/                # R scripts for data generation, models, analysis
│   ├── run_sims.R
│   ├── run_one_sim.R
|   ├── pscores_models.R
|   ├── outcome_models.R
│   ├── utils.R
│   └── plots.R
├── python_scripts/                   # Python scripts
│   ├── run_sims.py
│   ├── run_one_sim.py
|   ├── pscores_models.py
|   ├── outcome_models.py
│   ├── utils.py
│   └── plots.py
├── output/                   # Generated datasets and tables
└── figures/                  # Saved plots