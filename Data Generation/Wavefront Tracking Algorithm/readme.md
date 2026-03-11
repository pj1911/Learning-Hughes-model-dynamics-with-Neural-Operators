# Hughes Model Data Generation

This repository contains the code used to generate and preprocess data for the **Hughes model**.

## Overview

The main workflow is:

1. Run **`WFTcrowd_1.m`** in MATLAB to generate the raw data.
2. Run **`Matlab Data Preprocessing.py`** to preprocess the generated data for downstream use.

All other `.m` files in this folder are supporting/helper functions required by the main MATLAB script.

## Main Files

- **`WFTcrowd_1.m`** — main MATLAB script for data generation
- **`Matlab Data Preprocessing.py`** — Python script for preprocessing the generated MATLAB data before feeding to neural operators.

## Supporting MATLAB Files

The following files are helper functions used by `WFTcrowd_1.m`:

- `PdR.m`
- `callingfunc_1.m`
- `compute_psi_single.m`
- `id.m`
- `profil.m`
- `turningpoint.m`
- `unique_xy_coords.m`
