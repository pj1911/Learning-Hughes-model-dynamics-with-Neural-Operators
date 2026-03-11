# Hughes Model Dataset Generation

This folder provides reproducible pipelines to generate datasets for the Hughes pedestrian-flow model using two numerical backends:

1. Godunov (finite-volume transport + Eikonal solve per step)

2. Wavefront Tracking (WFT) (discontinuity-aware transport; samples reconstructed on a uniform grid)

Designed for our paper’s experiments and for benchmarking learning-based surrogates.

## Godunov Dataset Variants

This folder also contains multiple **Godunov-based dataset-generation settings** for the Hughes model. Each subfolder corresponds to a different boundary/initial-condition setup used to generate different types of datasets:

- **`Godunov Both exits closed`**  
  Scenario where pedestrians cannot exit from either end.

- **`Godunov Complex Case`**  
  Scenario with a more complex initial condition, containing a higher number of jumps.

- **`Godunov Easy Case`**  
  Scenario with a simpler initial condition, containing fewer jumps.

- **`Godunov Left exit closed`**  
  Scenario where the left exit is closed, so pedestrians can only exit from the right.

## Purpose

These cases are included to generate diverse Hughes-model datasets under different flow conditions. Together, they provide simple and complex benchmark settings for analysis, simulation, and downstream learning tasks used in the paper

## References

Wavefront Tracking (WFT): https://www-sop.inria.fr/members/Paola.Goatin/wft.html

