# Hughes Model 1D — WFT Data Generation

Pipeline for generating training data for the 1D Hughes crowd model using the Wavefront Tracking (WFT) algorithm.

---

## Files

| File | Role |
|------|------|
| `WFTcrowd.m` | Original WFT solver (hardcoded IC) |
| `PdR.m` | Riemann solver at each wavefront |
| `turningpoint.m` | Computes the turning point ξ |
| `id.m` | Rounds densities to the ε-grid |
| `profil.m` | Live density profile plotting |
| `reconstruct.m` | Converts WFT output → uniform (x,t) matrix |
| `run_a_sample.m` | Runs one IC through WFT + reconstruction, plots result |
| `generate_data.m` | Batch dataset generation |
| `plot_random_samples.m` | Visualises random samples from a saved dataset |

---

## The Model

The 1D Hughes model is the conservation law

$$\partial_t \rho + \partial_x \bigl(\rho\,(1-\rho)\,\text{sgn}(-\partial_x\phi)\bigr) = 0$$

where the cost potential φ satisfies $|\partial_x\phi| = c(\rho) = 1/(1-\rho)$ and the **turning point** ξ(t) is the point where pedestrians switch direction.  The domain is $x \in [-1,1]$, $t \in [0,T]$, with zero-density boundary conditions.

---

## Quick Start

```matlab
% 1. Run and plot a single sample (uses default IC from WFTcrowd)
run_a_sample()

% 2. Run with a custom initial condition
run_a_sample([-0.5, 0.3], [0.8, 0.4, 0.7], 3, 1/250)
%             ^ jump locs   ^ densities      T   ε

% 3. Generate a dataset
data = generate_data(1/250, 200, 3, [1 2 3 4 5], 1000, 250, 250);

% 4. Plot random training samples from the saved file
plot_random_samples('hughes_data_eps250_grid200_T3_5jumptypes.mat')
```

---

## Data Generation (`generate_data.m`)

```matlab
data = generate_data(epsilon, grid_size, T, n_jumps_vec, n_train, n_test, n_val)
```

| Argument | Example | Meaning |
|----------|---------|---------|
| `epsilon` | `1/250` | WFT mesh size (must be `1/N`, N ≥ 50) |
| `grid_size` | `200` | Output grid resolution (square) |
| `T` | `3` | Final simulation time |
| `n_jumps_vec` | `[1 2 3 4 5]` | Jump counts to include |
| `n_train/test/val` | `1000, 250, 250` | Samples per jump count per split |

**For each sample:**
1. Draw `n_j` jump locations uniformly from $(-1, 1)$ and `n_j+1` densities from $(0.02, 0.95)$.
2. Run the WFT solver (`PdR` + `turningpoint` + `id`).
3. **Reject** the sample if the cost-balance error exceeds 0.1 or the solver crashes.
4. Call `reconstruct` to evaluate the exact piecewise-constant WFT solution on a uniform `grid_size × grid_size` grid.
5. Build the pair $(X, Y)$:
   - $X$: `grid_size × grid_size`, initial condition in row 1, zeros elsewhere.
   - $Y$: `grid_size × grid_size`, full space-time solution $\rho(x,t)$.

The dataset is saved as `hughes_data_eps<N>_grid<G>_T<T>_<J>jumptypes.mat`.

---

## Grid Accuracy Error

> **Note:** this is *not* an error against an analytical ground truth.  It measures how much detail is lost purely due to grid resolution, by comparing two finite-resolution reconstructions of the same WFT solution.

Because the WFT solution is piecewise constant on *moving* intervals, the fixed grid introduces discretisation error wherever a wavefront passes through a grid cell interior.  To quantify this, each sample is reconstructed at **two resolutions from the same WFT data**:

- $\hat\rho$: target grid (`grid_size × grid_size`)
- $\tilde\rho$: finer grid (`2·grid_size × 2·grid_size`) — same solver output, finer sampling

The finer grid is block-averaged back to the target resolution:

$$\bar\rho_{i,j} = \frac{1}{4}\sum_{a=1}^{2}\sum_{b=1}^{2} \tilde\rho_{\,2(i-1)+a,\;2(j-1)+b}$$

The per-sample error is the **Mean Absolute Error (MAE)** between the two:

$$\text{MAE} = \frac{1}{N_t \cdot N_x}\sum_{i=1}^{N_t}\sum_{j=1}^{N_x} \bigl|\hat\rho_{i,j} - \bar\rho_{i,j}\bigr|$$

A low MAE means the target resolution is sufficient to represent the wavefronts without significant cell-straddling loss.  The dataset-wide average MAE is printed at the end of generation and stored in `data.meta.grid_mae`.  Typical values for a 200×200 grid are $O(10^{-4})$, and increase with jump count since more wavefronts means more cell-straddling events.

---

## Output Structure

```
data.train.X        % grid_size × grid_size × (n_train × |n_jumps_vec|)  single
data.train.Y        % same size — full WFT solution
data.train.n_jumps  % (n_train × |n_jumps_vec|) × 1  uint8

data.test.X / .Y / .n_jumps
data.val.X  / .Y / .n_jumps

data.meta           % epsilon, grid_size, T, n_jumps_vec, n_train/test/val, grid_mae
```
