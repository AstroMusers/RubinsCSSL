# RubinsCSSL
Rubin's repeated observations of cluster-scale strong lenses let us study their image structure and time delays. RubinsCSSL collects our simulations and analysis for these systems, with optional PyCS3 tools for fitting time delays.

## Installation

The PyCS3 time-delay fitting modules use the `time-delays` extra.

```bash
python -m pip install -e '.[time-delays]'
```

Set `RUBINSCSSL_PATH` to the checkout directory when the package is used from a different location. If it is unset, the package uses its checkout root. `data/` and `visuals/` are ignored by Git for local inputs and generated figures.

## Plotting

Install `.[plots]` for the corner and band-performance plots. These functions
and `make_population_corner_suite()` accept `typeplotback='dark'` for black
backgrounds with white axes. The default is white with black axes, or the
theme selected by an enclosing TDpy `plot_style()` block. Returned figures
and saved files share the same theme without changing later plots.
