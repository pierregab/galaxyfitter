# Paper reproduction coverage

This companion repository separates rerunning expensive fits from recomputing
statistics from saved likelihood maps. No new fits are needed for the latter.
Figure numbers refer to the manuscript snapshot used to assemble this repository.

| Result or component | Included material | Reproduction entry point |
| --- | --- | --- |
| Main projected calibration and slope readout | 89 complete 7 x 5 maps, configurations, metrics, family labels, reference summaries | `python scripts/reproduce.py` |
| Projected Figures 5–8 and 10 | Plotting code, map inputs and derived GravSphere2 profiles | `python scripts/reproduce.py --figures` |
| Matched 10k projected runs | Two complete maps, configurations and metrics | Used by the matched-comparison figure |
| A new projected fit | Original analytic engines, explicit paper launcher, one synthetic example | `python scripts/fit_projected.py examples/nfw_s3000 outputs/nfw_s3000 --seed 3000` (preview; add `--execute` to fit) |
| NFW, Burkert and gNFW mock generation | Original generators | `mocks/data_generation_*.py --help` |
| Truth/observed controls | Historical fitting engine and ensemble helper | Engine supplied; full ensemble inputs and exact campaign launch recipes are not bundled |
| Real-galaxy fits with photometry | Conditional-likelihood implementation and launcher options | Observational catalogues, prepared photometric inputs and complete real-galaxy campaign are not bundled |
| Other figures and manuscript | Not bundled | Refer to the paper |

The raw Gaia Challenge catalogues and full GravSphere2 chains are not bundled.
Their derived numerical outputs support the included reanalysis; rerunning those
fits needs the original inputs. Machine-specific paths in archived fit configs
are explicitly marked `<external:...>`, not presented as working local paths.

`python tests/check_release.py` checks complete grids, numeric agreement with
reference results, launcher/configuration parity and the distributed inventory.
These checks do not establish optimizer convergence or astrophysical validity.

The inference engines and mock generators are byte-identical to the archived
research sources. Only orchestration, imports/paths and publication packaging
are adapted where recorded in PROVENANCE.json. The research workspace itself
is not changed by this release.
