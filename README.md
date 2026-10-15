# Neural equilibrium DF inference for dwarf spheroidal galaxies

Companion code and numerical results for the paper *A neural equilibrium distribution-function
likelihood for line-of-sight kinematics of dwarf spheroidal galaxies: cusp/core
calibration and transfer*.

The code fits a neural equilibrium DF jointly with a spherical coreNFW potential.
The projected likelihood uses analytic integration and deterministic quadrature.
A 7 x 5 map in core strength and scale radius provides calibrated decision
statistics and density-slope readouts. Strong isotropic discrimination does not
imply a transferable cusp/core threshold; uncertainty calibration is population
and protocol dependent. See [the protocol and limitations](docs/PROTOCOL.md).

## Contents

- `fitters/projected.py`: frozen engine used for the main 2D-map campaign.
- `fitters/projected_resume.py`: checkpoint-enabled variant used for the 10k core.
- `scripts/fit_projected.py`: portable launcher with explicit paper settings.
- `fitters/six_coordinate.py`: historical engine for truth/observed controls.
- `mocks/data_generation_*.py`: original NFW, Burkert and gNFW mock generators.
- `scripts/legacy/run_calibration.py`: historical ensemble helper; its defaults
  are not the current paper protocol.
- `analysis/`: projected statistics and generators for paper Figures 5–8 and 10;
  statistical and profile helper functions extracted from their original modules.
- `data/`: small numerical maps/configurations for 89 1k and two matched 10k fits,
  plus derived GravSphere2 profile summaries. No model checkpoints or MCMC chains.
- `examples/nfw_s3000/`: one original synthetic catalogue, not a real galaxy.
- `reference/`: expected analysis output for regression checks.
- `PROVENANCE.json`: file checksums and source-to-release mapping.

The repository distributes the fitting method and the projected mock-map
analysis. [Reproduction coverage](docs/REPRODUCIBILITY.md) identifies which
paper results can be regenerated from the included files. Full regeneration of
the real-galaxy results and six-coordinate ensemble requires additional inputs.

## Which fitter?

| Paper data mode | Engine | Likelihood |
| --- | --- | --- |
| Truth: exact six coordinates | `fitters/six_coordinate.py` | Direct phase-space density |
| Observed: noisy six coordinates | `fitters/six_coordinate.py` | IWAE integration of measurement errors |
| Projected: LOS kinematics | `fitters/projected.py` | Analytic integration and deterministic quadrature |

These are different implementations used in different parts of the paper,
not duplicate copies. `projected_resume.py` is the checkpoint-enabled variant
of the projected engine. For current projected work use `scripts/fit_projected.py`;
the historical projected branch in `six_coordinate.py` is not the paper method.

## Install

Use **Python 3.12 or later** (the original plotting scripts use Python 3.12 syntax).
From this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Statistics and figures need NumPy, SciPy, pandas and Matplotlib. Training also
needs PyTorch. The engine chooses CUDA, then Apple MPS, then CPU; the original
campaign used MPS. Results can differ with device/library version. The compiled
cell accelerator is guarded by the engine and is not required on CPU.
`docs/environment-tested.json` records the environment used for release checks.

## Verify results without fitting

```sh
python scripts/reproduce.py
python tests/check_release.py
```

This recalculates the 89-run results and checks the original numerical summary,
map completeness, source hashes and launcher/configuration parity. It does not
run training or use the GPU. Generated summaries are written under `data/`.

To rebuild Figures 5–8 and 10, put `pdflatex` (with NewTX fonts) and `pdftoppm`
(Poppler) on PATH, then run:

```sh
python scripts/reproduce.py --figures
```

Outputs: `outputs/figures/{pdf,png,pgf,captions}`. These outputs are Git-ignored.

## Optional fitting

Preview the exact command (no training):

```sh
python scripts/fit_projected.py examples/nfw_s3000 outputs/nfw_s3000 --seed 3000
```

Add `--execute` to train. Add `--resume` **from the first invocation** to use the
checkpoint-enabled engine. Both switches are explicit:

```sh
python scripts/fit_projected.py examples/nfw_s3000 outputs/nfw_s3000 --seed 3000 --resume --execute
```

Log: `outputs/nfw_s3000/fit.log`. With the checkpoint engine, request a stop by
creating `outputs/nfw_s3000/STOP`; it exits at a supported checkpoint boundary,
not immediately. Remove STOP and rerun the identical command to resume. Early
training may still need to be repeated before the first checkpoint. Completed
outputs and non-checkpoint runs are protected against accidental overwriting.

Use `--sample 10k` for the reported 10k epoch schedule; this does not change the
input catalogue size. Photometry options are described in `docs/PROTOCOL.md`.
A full fit is computationally expensive. Use the launcher to apply the paper
settings; the engine defaults differ from the reported protocol.

A new synthetic catalogue can be generated independently, for example:

```sh
python mocks/data_generation_nfw.py --output-dir outputs/new_mock --seed 3000 --n-stars 1000 --noise-preset learnable
```

## License and citation

Code and documentation: [MIT](LICENSE). Original research data and synthetic
examples: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See [attribution and scope](docs/LICENSE_NOTES.md).

[CITATION.cff](CITATION.cff) supplies the software citation metadata. Please cite
the associated paper when using the method or research outputs.
