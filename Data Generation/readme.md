Hughes Model Dataset Generation:

This subfolder provides reproducible pipelines to generate datasets for the Hughes pedestrian-flow model using two numerical backends:

1. Godunov (finite-volume transport + Eikonal solve per step)

2. Wavefront Tracking (WFT) (discontinuity-aware transport; samples reconstructed on a uniform grid)

Designed for our paper’s experiments and for benchmarking learning-based surrogates.
