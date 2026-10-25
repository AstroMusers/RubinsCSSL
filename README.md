# RubinsCSSL
Rubin's repeated observations of cluster-scale strong lenses let us study their image structure and time delays. RubinsCSSL collects our simulations and analysis for these systems, with optional PyCS3 tools for fitting time delays.

## Installation

The PyCS3 time-delay fitting modules use the `time-delays` extra.

```bash
python -m pip install -e '.[time-delays]'
```

Set `RUBINSCSSL_PATH` to the checkout directory when the package is used from a different location. If it is unset, the package uses its checkout root. `data/` and `visuals/` are ignored by Git for local inputs and generated figures.

## Plotting

Install `.[plots]` for the corner, static band-performance, and interactive
Bokeh plots.

```bash
python -m pip install -e '.[plots]'
```

`plot_band_performance()` and `plot_band_performance_bokeh()` render the same
assessability, conditional-accuracy, and end-to-end fractions from the summary
table. The HTML version adds hover values and a legend that can hide series.
Both plotting functions and `make_population_corner_suite()` accept
`typeplotback='dark'` for black backgrounds with white axes. The default is
white with black axes, or the theme selected by an enclosing TDpy
`plot_style()` block. Returned figures and saved files share the selected theme
without changing later plots.

## Example

The [synthetic feature-pipeline notebook](examples/synthetic_feature_pipeline.ipynb)
passes explicitly toy dual-image curves and toy fit assessments through
`build_population_analysis_tables()`. It generates static and Bokeh band
summaries under the ignored `visuals/` directory. The synthetic example does
not produce Rubin observations or PyCS3 fits and measures no time-delay recovery
performance. It runs quickly and needs no PyCS3 installation.
