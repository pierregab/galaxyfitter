# Scope and scientific protocol

This is a compact code release, not the full research workspace. Its default
projected protocol is the manuscript's **7 x 5 map in (n, Rs)**, not the earlier
one-dimensional scan, the G gate, or a gamma-coordinate experiment.

The fitting engines are byte-identical copies of the files that produced the
reported campaign. Their argparse defaults are historical and are NOT the
paper's protocol. Use `scripts/fit_projected.py`.

## Projected likelihood and map

- coreNFW potential, neural equilibrium DF f(epsilon, L), hidden width 48.
- Analytic azimuthal integration and Gaussian error convolution; deterministic
  depth quadrature. No Monte-Carlo likelihood noise at a fixed model state.
- Gauss-Legendre density/moment rule: 96 speed by 32 polar nodes.
- 1k joint training: 250 warm-up / 800 joint epochs; batch size 500.
- Scan: n = -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1; Rs/Rhalf = 2, 4, 8, 16, 32.
  100 adaptation epochs per row, 200 epochs per node.
- Dispersion and VSP weights: 10 and 5. Strong rc/Rs shape-prior weights: zero.
  Weak potential prior, parameter boxes, initialization and anchors remain.
- 10k: warm-up/joint epochs 60/200; scan epochs and row adaptation remain 200/100.
- Checkpoints are selected by the composite training objective; the reported
  decision statistics use pure NLL. This is a regularized, finite-budget profile,
  not a demonstration of globally minimized likelihood.

Original details (including the start-node sigmoid convention and boundaries)
remain in the engine. We have not silently corrected algorithms for the release.
Each reference `fit_config.json` retains scientific fields; machine-specific
input/output paths are replaced by explicit placeholders.

## Readout

`analysis/paper_numbers_grid2d.py` reads the 2D CSV, flags nodes more than three
NLL units above the lower convex hull in each row, and computes P(n) as the
minimum **pure NLL** across Rs. The hull filter is a fixed shape heuristic;
profile likelihoods need not be convex. Do not substitute the fitter's 1D
CSV, which chooses Rs using the composite objective.

- dNLL = P(1) - P(-0.5), core below the calibrated threshold.
- A_asym: difference of trapezoidal areas of P(n)-min P on nodes n <= 0.5
  and n > 0.5. No integral crosses the cut; no branch-length normalization.
  Core above the calibrated threshold.
- Local slope: exp(-Delta NLL) weighted mean over retained map nodes.
  These discrete weights are not a Bayesian posterior.
- The interval threshold is empirically fitted on the mocks. Its transfer to
  other families and to conditional real-galaxy fits is not guaranteed.

Thresholds are recomputed from the 60 isotropic calibration maps. All 89 maps
contribute to the slope and interval assessment. The nine Gaia catalogues
share two parent systems; they are not nine independent physical systems.
GravSphere2 comparison values are provided as derived comparison inputs, not
recomputed from its full MCMC chains in this release.

## Input data and photometry

The engine expects `draco_nfw_observations.csv`, with columns:
RA, Dec (degrees), Dist (kpc), pmRA, pmDec (mas/yr), v_los (km/s), and e_RA,
e_Dec, e_Dist, e_pmRA, e_pmDec, e_v_los in the corresponding units.
The historical loader expects all columns even in LOS mode; proper motions and
individual distances are not kinematic observations in that likelihood.
Use the provided generator/example for an input-format reference. Real inputs
also require the correct geometry/systemic metadata; do not insert arbitrary
coordinates into the Draco mock defaults.

`--photometry PATH` forwards the engine's photometry option, enabling its
conditional velocity likelihood. Use the engine's documented table format
(radius in kpc, surface density, uncertainty). `--init-from RUN_DIR` loads the
joint model state; `--systemic-from-data` is available for real-galaxy inputs.
No external observational catalogue or complete real-galaxy preparation pipeline
is redistributed here. The conditional likelihood has not been calibrated by
the 60 isotropic joint-likelihood mocks.

## Six-coordinate controls

`fitters/six_coordinate.py` and `scripts/legacy/run_calibration.py` retain the historical scientific implementation for truth/observed controls. They use a different numerical
implementation from the projected analytic engine. The legacy launcher's
current defaults are projected: do not use it as a shortcut to reproduce a
truth/observed campaign without the corresponding archived run configuration.
This compact release does not reproduce Figure 4 or all real-galaxy figures.
