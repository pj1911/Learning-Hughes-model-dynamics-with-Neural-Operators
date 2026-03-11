# Neural operators struggle to learn complex PDEs in pedestrian mobility: Hughes model case study
https://doi.org/10.1016/j.ait.2025.100005

This paper investigates the limitations of neural operators in learning solutions for the Hughes model, a first-order hyperbolic conservation law system for crowd dynamics. The model couples a Fokker–Planck equation, representing pedestrian density, with a Hamilton–Jacobi type equation (the eikonal equation). In this study, we assess the performance of three state of the art neural operators Fourier Neural Operator (FNO), Wavelet Neural Operator (WNO), and Multiwavelet Neural Operator (MWNO) in various challenging scenarios. Our results show that these neural operators perform well in easy scenarios characterized by fewer discontinuities in the initial condition, but, they struggle in complex scenarios with multiple initial discontinuities and dynamic boundary condition. The predicted solutions often appear smoother, resulting in a reduction in total variation and a loss of important physical features. This smoothing behavior is similar to issues discussed in Daganzo (1995), where models that introduce artificial diffusion were shown to miss essential features such as shock waves in hyperbolic systems. This suggests that current neural operator architectures may introduce unintended regularization effects that limit their ability to capture transport dynamics governed by discontinuities. Since the Hughes model shares important structural features with models used in traffic flow, these results also raise concerns about the ability of neural operator architectures to generalize to traffic applications where shock preservation is essential.

<img width="929" height="473" alt="Screenshot 2025-09-26 at 2 01 07 PM" src="https://github.com/user-attachments/assets/5d9d09c5-39c3-4a5d-a8a4-c5a3fee30df6" />

## Repository Overview

This GitHub contains the **data generation, training, and testing code** for the above paper. It includes reproducible pipelines for generating Hughes model datasets using both **Godunov** and **Wavefront Tracking (WFT)** solvers, along with code for preprocessing the generated data and training/testing neural operator models on these datasets.

The repository covers multiple Hughes-model settings, including **easy** and **complex** initial conditions, as well as different **boundary-condition cases** such as **both exits closed** and **left exit closed**. These cases are designed to evaluate how well neural operators handle transport-dominated PDE dynamics under varying levels of difficulty, especially in the presence of **discontinuities**, **shock-like structures**, and **dynamic boundaries**.

The learning experiments in this repository focus on three neural operator architectures:

- **Fourier Neural Operator (FNO)**
- **Wavelet Neural Operator (WNO)**
- **Multiwavelet Neural Operator (MWNO)**

Overall, this repository supports the full workflow of the paper:  
**dataset generation → preprocessing → model training → model testing/evaluation**, with an emphasis on understanding why neural operators struggle to preserve important physical features in complex Hughes-model scenarios.
