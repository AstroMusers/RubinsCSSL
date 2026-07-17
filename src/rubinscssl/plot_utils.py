from __future__ import annotations

import corner
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

def _prepare_corner_frame(
    dataframe: pd.DataFrame,
    *,
    features: Sequence[str],
    required_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """Clean a corner-plot frame and drop zero-range dimensions."""
    columns = list(features) + list(required_columns)
    frame = (
        dataframe[columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    usable_features = [
        feature
        for feature in features
        if frame[feature].nunique(dropna=True) >= 2
    ]
    if not usable_features:
        raise ValueError(
            "None of the requested corner-plot features has finite "
            "non-zero dynamic range."
        )
    return frame, usable_features


def _corner_ranges(values: np.ndarray) -> list[tuple[float, float]]:
    """Return padded finite ranges for corner.py."""
    ranges = []
    for column in range(values.shape[1]):
        finite = values[:, column][np.isfinite(values[:, column])]
        lower = float(np.min(finite))
        upper = float(np.max(finite))
        if upper == lower:
            pad = 0.5 if lower == 0.0 else 0.05 * abs(lower)
        else:
            pad = 0.03 * (upper - lower)
        ranges.append((lower - pad, upper + pad))
    return ranges


def plot_binary_corner(
    dataframe: pd.DataFrame,
    *,
    features: Sequence[str],
    labels: Sequence[str],
    outcome: str,
    false_label: str,
    true_label: str,
    title: str,
    filename: str | Path | None = None,
    bins: int = 20,
    smooth: float | None = 1.0,
    minimum_rows_per_group: int = 2,
):
    """Overlay feature distributions for a binary outcome.

    Constant or fully missing dimensions are removed automatically. The
    returned dictionary records which dimensions and groups were plotted.
    """
    if len(features) != len(labels):
        raise ValueError("features and labels must have equal lengths.")

    label_map = dict(zip(features, labels))
    frame, usable_features = _prepare_corner_frame(
        dataframe,
        features=features,
        required_columns=[outcome],
    )
    usable_labels = [label_map[item] for item in usable_features]

    frame[outcome] = frame[outcome].astype(bool)
    false_values = frame.loc[
        ~frame[outcome], usable_features
    ].to_numpy(dtype=float)
    true_values = frame.loc[
        frame[outcome], usable_features
    ].to_numpy(dtype=float)

    combined = frame[usable_features].to_numpy(dtype=float)
    ranges = _corner_ranges(combined)

    figure = None
    legend_handles = []

    if len(false_values) >= minimum_rows_per_group:
        figure = corner.corner(
            false_values,
            labels=usable_labels,
            bins=bins,
            range=ranges,
            smooth=smooth,
            smooth1d=smooth,
            color="C1",
            plot_datapoints=True,
            plot_density=False,
            plot_contours=True,
            fill_contours=False,
            quiet=True,
            hist_kwargs={"alpha": 0.55},
            data_kwargs={"alpha": 0.20, "ms": 2},
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="C1",
                label=f"{false_label} (N={len(false_values)})",
            )
        )

    if len(true_values) >= minimum_rows_per_group:
        figure = corner.corner(
            true_values,
            fig=figure,
            labels=usable_labels,
            bins=bins,
            range=ranges,
            smooth=smooth,
            smooth1d=smooth,
            color="C0",
            plot_datapoints=True,
            plot_density=False,
            plot_contours=True,
            fill_contours=False,
            quiet=True,
            hist_kwargs={"alpha": 0.55},
            data_kwargs={"alpha": 0.20, "ms": 2},
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="C0",
                label=f"{true_label} (N={len(true_values)})",
            )
        )

    if figure is None:
        raise ValueError(
            f"Neither outcome group has at least "
            f"{minimum_rows_per_group} complete rows."
        )

    figure.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
    )
    figure.suptitle(title, fontsize=15, y=1.01)

    if filename is not None:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, bbox_inches="tight", dpi=200)

    return {
        "figure": figure,
        "features_used": usable_features,
        "n_false": len(false_values),
        "n_true": len(true_values),
    }


def plot_continuous_error_corner(
    dataframe: pd.DataFrame,
    *,
    features: Sequence[str],
    labels: Sequence[str],
    error_feature: str = "log10_abs_delay_error_plus",
    error_label: str = (
        r"$\log_{10}(|\widehat{\Delta t}-\Delta t_{\rm true}|)$"
    ),
    title: str,
    filename: str | Path | None = None,
    bins: int = 20,
    smooth: float | None = 1.0,
):
    """Make a corner plot that includes a continuous delay-error axis."""
    if len(features) != len(labels):
        raise ValueError("features and labels must have equal lengths.")

    all_features = list(features) + [error_feature]
    all_labels = list(labels) + [error_label]
    label_map = dict(zip(all_features, all_labels))

    frame, usable_features = _prepare_corner_frame(
        dataframe,
        features=all_features,
    )
    if error_feature not in usable_features:
        raise ValueError(
            f"{error_feature!r} has no finite non-zero dynamic range."
        )

    usable_labels = [label_map[item] for item in usable_features]
    values = frame[usable_features].to_numpy(dtype=float)

    figure = corner.corner(
        values,
        labels=usable_labels,
        bins=bins,
        range=_corner_ranges(values),
        smooth=smooth,
        smooth1d=smooth,
        show_titles=True,
        title_quantiles=[0.16, 0.50, 0.84],
        quantiles=[0.16, 0.50, 0.84],
        plot_datapoints=True,
        plot_density=True,
        plot_contours=True,
        quiet=True,
    )
    figure.suptitle(title, fontsize=15, y=1.01)

    if filename is not None:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, bbox_inches="tight", dpi=200)

    return {
        "figure": figure,
        "features_used": usable_features,
        "n_rows": len(frame),
    }


def plot_band_performance(
    band_summary: pd.DataFrame,
    *,
    curve_type: str,
    estimator: str,
    filename: str | Path | None = None,
):
    """Plot assessability, conditional accuracy, and end-to-end accuracy."""
    data = band_summary.loc[
        (band_summary["curve_type"] == curve_type)
        & (band_summary["estimator"] == estimator)
    ].copy()

    if data.empty:
        raise ValueError(
            f"No band-summary rows for {curve_type=}, {estimator=}."
        )

    data = data.set_index("band").reindex(["g", "r", "i"])
    columns = [
        "assessable_fraction_given_attempted",
        "accuracy_fraction_given_assessable",
        "end_to_end_accurate_fraction",
    ]
    labels = [
        "Assessable | attempted",
        "Accurate | assessable",
        "End-to-end accurate",
    ]

    axis = data[columns].plot(kind="bar", figsize=(9, 5))
    axis.set_xlabel("LSST band")
    axis.set_ylabel("Fraction")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(
        f"{curve_type.capitalize()} {estimator} performance by band"
    )
    axis.legend(labels)
    axis.figure.tight_layout()

    if filename is not None:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        axis.figure.savefig(filename, bbox_inches="tight", dpi=200)

    return axis.figure


def make_population_corner_suite(
    analysis_table: pd.DataFrame,
    *,
    curve_type: str,
    estimator: str,
    output_directory: str | Path,
    features: Sequence[str],
    labels: Sequence[str],
    bands: Sequence[str] = ("g", "r", "i"),
    minimum_rows: int = 5,
) -> dict[str, dict]:
    """Create global and per-band corner plots for one analysis branch.

    Assessability is plotted among attempted measurements. Accuracy is plotted
    only among assessable measurements. End-to-end usability is plotted among
    available light curves and therefore retains model-gate failures in the
    denominator.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    selected = analysis_table.loc[
        (analysis_table["curve_type"] == curve_type)
        & (analysis_table["estimator"] == estimator)
    ].copy()

    outputs: dict[str, dict] = {}

    attempted = selected.loc[selected["attempted"]].copy()
    if len(attempted) >= minimum_rows:
        try:
            outputs["global_assessability"] = plot_binary_corner(
                attempted,
                features=features,
                labels=labels,
                outcome="assessable",
                false_label="Not assessable",
                true_label="Assessable",
                title=(
                    f"{curve_type.capitalize()} {estimator}: "
                    "assessability among attempted measurements"
                ),
                filename=output_directory
                / f"{curve_type}_{estimator}_assessability_corner.png",
            )
        except ValueError as exc:
            outputs["global_assessability"] = {"skipped": str(exc)}

    assessable = attempted.loc[attempted["assessable"]].copy()
    if len(assessable) >= minimum_rows:
        try:
            outputs["global_accuracy"] = plot_binary_corner(
                assessable,
                features=features,
                labels=labels,
                outcome="accurate",
                false_label="Inaccurate",
                true_label="Accurate",
                title=(
                    f"{curve_type.capitalize()} {estimator}: "
                    "accuracy among assessable measurements"
                ),
                filename=output_directory
                / f"{curve_type}_{estimator}_accuracy_corner.png",
            )
        except ValueError as exc:
            outputs["global_accuracy"] = {"skipped": str(exc)}

        try:
            outputs["global_continuous_error"] = (
                plot_continuous_error_corner(
                    assessable,
                    features=features,
                    labels=labels,
                    title=(
                        f"{curve_type.capitalize()} {estimator}: "
                        "continuous delay-error relationships"
                    ),
                    filename=output_directory
                    / f"{curve_type}_{estimator}_continuous_error_corner.png",
                )
            )
        except ValueError as exc:
            outputs["global_continuous_error"] = {"skipped": str(exc)}

    available = selected.loc[selected["available"]].copy()
    if len(available) >= minimum_rows:
        try:
            outputs["global_end_to_end"] = plot_binary_corner(
                available,
                features=features,
                labels=labels,
                outcome="usable",
                false_label="No accurate usable delay",
                true_label="Accurate usable delay",
                title=(
                    f"{curve_type.capitalize()} {estimator}: "
                    "end-to-end recovery"
                ),
                filename=output_directory
                / f"{curve_type}_{estimator}_end_to_end_corner.png",
            )
        except ValueError as exc:
            outputs["global_end_to_end"] = {"skipped": str(exc)}

    for band in bands:
        band_attempted = attempted.loc[attempted["band"] == band]
        if len(band_attempted) >= minimum_rows:
            try:
                outputs[f"{band}_assessability"] = plot_binary_corner(
                    band_attempted,
                    features=features,
                    labels=labels,
                    outcome="assessable",
                    false_label="Not assessable",
                    true_label="Assessable",
                    title=(
                        f"{band}-band {curve_type} {estimator}: "
                        "assessability"
                    ),
                    filename=output_directory
                    / f"{band}_{curve_type}_{estimator}_assessability_corner.png",
                )
            except ValueError as exc:
                outputs[f"{band}_assessability"] = {"skipped": str(exc)}

        band_assessable = assessable.loc[assessable["band"] == band]
        if len(band_assessable) >= minimum_rows:
            try:
                outputs[f"{band}_accuracy"] = plot_binary_corner(
                    band_assessable,
                    features=features,
                    labels=labels,
                    outcome="accurate",
                    false_label="Inaccurate",
                    true_label="Accurate",
                    title=(
                        f"{band}-band {curve_type} {estimator}: "
                        "accuracy"
                    ),
                    filename=output_directory
                    / f"{band}_{curve_type}_{estimator}_accuracy_corner.png",
                )
            except ValueError as exc:
                outputs[f"{band}_accuracy"] = {"skipped": str(exc)}

            try:
                outputs[f"{band}_continuous_error"] = (
                    plot_continuous_error_corner(
                        band_assessable,
                        features=features,
                        labels=labels,
                        title=(
                            f"{band}-band {curve_type} {estimator}: "
                            "continuous delay error"
                        ),
                        filename=output_directory
                        / f"{band}_{curve_type}_{estimator}_continuous_error_corner.png",
                    )
                )
            except ValueError as exc:
                outputs[f"{band}_continuous_error"] = {
                    "skipped": str(exc)
                }

    return outputs