# RubinsCSSL
Rubin's repeated observations of cluster-scale strong lenses let us study their image structure and time delays. RubinsCSSL collects our simulations and analysis for these systems, with optional PyCS3 tools for fitting time delays.

## Installation

Use Python 3.13 or newer. Install the package with the interpreter that will run
the analysis and tests.

```bash
python -m pip install -e .
```

The PyCS3 time-delay fitting modules use the `time-delays` extra.

```bash
python -m pip install -e '.[time-delays]'
```

Set `RUBINSCSSL_PATH` to the checkout directory when the package is used from a different location. If it is unset, the package uses its checkout root. `data/` and `visuals/` are ignored by Git for local inputs and generated figures.

## Tests

Install the test, plotting, and time-delay dependencies before running the full
suite.

```bash
python -m pip install -e '.[dev,plots,time-delays]'
python -m pytest
```

The compiled Rubin Science Pipelines are a separate environment prerequisite.
Their integration check is skipped when those modules are unavailable. SLSim is
a required package dependency, so a missing SLSim installation remains an error.

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
builds model and noisy observed light curves for six toy lens systems in the
`g`, `r`, and `i` bands. It supplies illustrative fit outcomes, then summarizes
feature availability, assessability, delay accuracy, and error by band. The
static plots need no PyCS3 installation. Install the optional `plots` extra for
the Bokeh summary. These fixtures exercise the pipeline and provide no estimate
of Rubin time-delay recovery.
