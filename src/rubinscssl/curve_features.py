"""Shared array conversion and summary helpers for light-curve tables."""

import json
import pickle
from pathlib import Path

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _finite_float(value: Any, default: float = np.nan) -> float:
    """Return a finite scalar float when possible."""
    try:
        array = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return default
    if array.size == 0:
        return default
    value = float(array[0])
    return value if np.isfinite(value) else default


def _nullable_bool(value: Any):
    """Normalize a value to True, False, or pandas.NA."""
    if value is None or value is pd.NA:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    return bool(value)


def _curve_arrays(light_curve: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read finite, time-sorted arrays with positive magnitude uncertainties."""
    times = np.asarray(light_curve.jds, dtype=float)
    magnitudes = np.asarray(light_curve.mags, dtype=float)
    errors = np.asarray(light_curve.magerrs, dtype=float)

    if not (len(times) == len(magnitudes) == len(errors)):
        raise ValueError(
            "Light-curve arrays have inconsistent lengths: "
            f"{len(times)}, {len(magnitudes)}, {len(errors)}."
        )

    valid = (
        np.isfinite(times)
        & np.isfinite(magnitudes)
        & np.isfinite(errors)
        & (errors > 0.0)
    )
    order = np.argsort(times[valid])
    return times[valid][order], magnitudes[valid][order], errors[valid][order]


def _summarize_curve(light_curve: Any) -> dict[str, Any]:
    """Summarize one light curve without applying a scientific cut."""
    times, magnitudes, errors = _curve_arrays(light_curve)
    n_points = int(len(times))

    if n_points == 0:
        return {
            "n": 0,
            "first_epoch": np.nan,
            "last_epoch": np.nan,
            "baseline": np.nan,
            "median_cadence": np.nan,
            "max_gap": np.nan,
            "median_error": np.nan,
            "mean_error": np.nan,
            "median_magnitude": np.nan,
        }

    if n_points == 1:
        baseline = 0.0
        median_cadence = np.nan
        max_gap = np.nan
    else:
        gaps = np.diff(times)
        baseline = float(times[-1] - times[0])
        median_cadence = float(np.median(gaps))
        max_gap = float(np.max(gaps))

    return {
        "n": n_points,
        "first_epoch": float(times[0]),
        "last_epoch": float(times[-1]),
        "baseline": baseline,
        "median_cadence": median_cadence,
        "max_gap": max_gap,
        "median_error": float(np.median(errors)),
        "mean_error": float(np.mean(errors)),
        "median_magnitude": float(np.median(magnitudes)),
    }


def build_light_curve_feature_table_from_lcs(
    lens_systems: Mapping,
    *,
    bands: Sequence[str] = ("r", "g", "i"),
) -> pd.DataFrame:
    """Build one feature row per system, band, and curve type.

    This function matches the actual structure of ``dual_lenses_with_lcs`` in
    the supplied notebook. It reads ``system[band]["lcs"]`` for observed
    curves and ``system[band]["model_lcs"]`` for ideal-model curves.

    Systems and bands that cannot provide curves are retained with a
    non-success ``feature_status`` so that they remain visible in denominators.
    """
    rows: list[dict[str, Any]] = []

    for system_id, system in lens_systems.items():
        system_status = system.get("system_status", "unknown")
        true_time_delays = np.asarray(
            system.get("time_delay", []), dtype=float
        ).ravel()
        true_delay = (
            float(abs(true_time_delays[0] - true_time_delays[1]))
            if true_time_delays.size >= 2
            else np.nan
        )

        metadata = {
            "system_id": system_id,
            "system_status": system_status,
            "batch": system.get("batch", np.nan),
            "z_lens": _finite_float(system.get("z_lens", np.nan)),
            "z_source": _finite_float(system.get("z_source", np.nan)),
            "einstein_radius": _finite_float(
                system.get("einstein_radius", np.nan)
            ),
            "true_delay_feature": true_delay,
        }

        for band in bands:
            band_record = system.get(band, {})
            band_status = band_record.get("band_status", "band_unavailable")

            for curve_type, curve_key in (
                ("model", "model_lcs"),
                ("observed", "lcs"),
            ):
                row = {
                    **metadata,
                    "band": str(band),
                    "curve_type": curve_type,
                    "band_status": band_status,
                    "feature_status": "not_started",
                }

                curves = band_record.get(curve_key)
                if system_status != "success":
                    row["feature_status"] = "system_unavailable"
                    rows.append(row)
                    continue
                if band_status != "success" or curves is None:
                    row["feature_status"] = str(band_status)
                    rows.append(row)
                    continue
                if not isinstance(curves, (list, tuple)) or len(curves) != 2:
                    row["feature_status"] = "invalid_curve_pair"
                    rows.append(row)
                    continue

                try:
                    stats_a = _summarize_curve(curves[0])
                    stats_b = _summarize_curve(curves[1])
                    times_a, _, _ = _curve_arrays(curves[0])
                    times_b, _, _ = _curve_arrays(curves[1])
                except Exception as exc:
                    row.update(
                        feature_status="feature_extraction_failed",
                        feature_exception_type=type(exc).__name__,
                        feature_exception_message=str(exc),
                    )
                    rows.append(row)
                    continue

                if len(times_a) and len(times_b):
                    overlap_start = max(times_a[0], times_b[0])
                    overlap_end = min(times_a[-1], times_b[-1])
                    overlap_baseline = float(
                        max(0.0, overlap_end - overlap_start)
                    )
                    shared_epochs = int(
                        np.intersect1d(times_a, times_b).size
                    )
                else:
                    overlap_baseline = 0.0
                    shared_epochs = 0

                n_a = stats_a["n"]
                n_b = stats_b["n"]
                n_min = min(n_a, n_b)
                n_max = max(n_a, n_b)
                n_total = n_a + n_b

                cadence_values = np.asarray(
                    [stats_a["median_cadence"], stats_b["median_cadence"]],
                    dtype=float,
                )
                cadence_values = cadence_values[np.isfinite(cadence_values)]

                gap_values = np.asarray(
                    [stats_a["max_gap"], stats_b["max_gap"]],
                    dtype=float,
                )
                gap_values = gap_values[np.isfinite(gap_values)]

                error_values = np.asarray(
                    [stats_a["median_error"], stats_b["median_error"]],
                    dtype=float,
                )
                error_values = error_values[np.isfinite(error_values)]

                baseline_values = np.asarray(
                    [stats_a["baseline"], stats_b["baseline"]],
                    dtype=float,
                )
                baseline_values = baseline_values[np.isfinite(baseline_values)]

                row.update(
                    feature_status="success",
                    n_A=n_a,
                    n_B=n_b,
                    n_min=n_min,
                    n_max=n_max,
                    n_total=n_total,
                    n_shared=shared_epochs,
                    shared_fraction=(
                        shared_epochs / n_max if n_max > 0 else 0.0
                    ),
                    first_epoch_A=stats_a["first_epoch"],
                    first_epoch_B=stats_b["first_epoch"],
                    last_epoch_A=stats_a["last_epoch"],
                    last_epoch_B=stats_b["last_epoch"],
                    baseline_A=stats_a["baseline"],
                    baseline_B=stats_b["baseline"],
                    baseline_min=(
                        float(np.min(baseline_values))
                        if baseline_values.size
                        else np.nan
                    ),
                    overlap_baseline=overlap_baseline,
                    median_cadence_A=stats_a["median_cadence"],
                    median_cadence_B=stats_b["median_cadence"],
                    median_cadence=(
                        float(np.median(cadence_values))
                        if cadence_values.size
                        else np.nan
                    ),
                    max_gap_A=stats_a["max_gap"],
                    max_gap_B=stats_b["max_gap"],
                    max_gap=(
                        float(np.max(gap_values))
                        if gap_values.size
                        else np.nan
                    ),
                    median_error_A=stats_a["median_error"],
                    median_error_B=stats_b["median_error"],
                    median_error=(
                        float(np.median(error_values))
                        if error_values.size
                        else np.nan
                    ),
                    mean_error_A=stats_a["mean_error"],
                    mean_error_B=stats_b["mean_error"],
                    median_magnitude_A=stats_a["median_magnitude"],
                    median_magnitude_B=stats_b["median_magnitude"],
                    median_image_contrast=(
                        stats_b["median_magnitude"]
                        - stats_a["median_magnitude"]
                    ),
                    points_per_100_days=(
                        100.0 * n_min / overlap_baseline
                        if overlap_baseline > 0.0
                        else np.nan
                    ),
                    delay_to_overlap_ratio=(
                        abs(true_delay) / overlap_baseline
                        if overlap_baseline > 0.0
                        and np.isfinite(true_delay)
                        else np.nan
                    ),
                )
                rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=["system_id", "band", "curve_type", "feature_status"])
    duplicated = table.duplicated(
        ["system_id", "band", "curve_type"], keep=False
    )
    if duplicated.any():
        examples = table.loc[
            duplicated, ["system_id", "band", "curve_type"]
        ].head().to_dict("records")
        raise ValueError(
            "Feature table is not unique on system_id, band, curve_type. "
            f"Examples: {examples}"
        )
    return table


def flatten_baseline_results(
    baseline_results: Sequence[Mapping[str, Any]],
    *,
    estimator: str,
    curve_types: Sequence[str] = ("model", "observed"),
) -> pd.DataFrame:
    """Convert the notebook's nested baseline-result rows to long form.

    The notebook stores dictionaries such as ``model_spline_results`` and
    ``observed_spline_assessment`` inside each system-band row. This function
    produces one flat row for each curve type.
    """
    rows: list[dict[str, Any]] = []

    for source_row in baseline_results:
        true_delay = _finite_float(source_row.get("true_delay", np.nan))

        model_assessment = source_row.get(
            f"model_{estimator}_assessment", {}
        )
        model_gate_accurate = _nullable_bool(
            model_assessment.get("accurate", pd.NA)
            if isinstance(model_assessment, Mapping)
            else pd.NA
        )

        for curve_type in curve_types:
            result = source_row.get(
                f"{curve_type}_{estimator}_results", {}
            )
            assessment = source_row.get(
                f"{curve_type}_{estimator}_assessment", {}
            )
            status = source_row.get(
                f"{curve_type}_status", "not_recorded"
            )

            result = result if isinstance(result, Mapping) else {}
            assessment = (
                assessment if isinstance(assessment, Mapping) else {}
            )

            attempted = bool(result)
            converged = bool(result.get("converged", False)) if attempted else False
            assessable = _nullable_bool(
                assessment.get("assessable", converged)
                if attempted
                else False
            )

            estimated_delay = _finite_float(
                result.get("delay", np.nan)
            )
            delay_error = _finite_float(
                assessment.get("delay_error", np.nan)
            )
            if (
                not np.isfinite(delay_error)
                and np.isfinite(estimated_delay)
                and np.isfinite(true_delay)
            ):
                delay_error = estimated_delay - true_delay

            absolute_delay_error = _finite_float(
                assessment.get("absolute_delay_error", np.nan)
            )
            if (
                not np.isfinite(absolute_delay_error)
                and np.isfinite(delay_error)
            ):
                absolute_delay_error = abs(delay_error)

            relative_delay_error = _finite_float(
                assessment.get("relative_delay_error", np.nan)
            )
            if (
                not np.isfinite(relative_delay_error)
                and np.isfinite(absolute_delay_error)
                and np.isfinite(true_delay)
                and true_delay != 0.0
            ):
                relative_delay_error = absolute_delay_error / abs(true_delay)

            assessable_bool = (
                False if assessable is pd.NA else bool(assessable)
            )
            accurate = (
                _nullable_bool(assessment.get("accurate", pd.NA))
                if assessable_bool
                else pd.NA
            )

            row = {
                "system_id": source_row.get("system_id"),
                "band": source_row.get("band"),
                "curve_type": curve_type,
                "estimator": estimator,
                "true_delay": true_delay,
                "status": status,
                "attempted": attempted,
                "converged": converged,
                "assessable": assessable_bool,
                "accurate": accurate,
                "estimated_delay": estimated_delay,
                "delay_error": delay_error,
                "abs_delay_error": absolute_delay_error,
                "relative_delay_error": relative_delay_error,
                "accuracy_tolerance": _finite_float(
                    assessment.get("tolerance", np.nan)
                ),
                "timeshift_A": _finite_float(
                    result.get("timeshift_A", np.nan)
                ),
                "timeshift_B": _finite_float(
                    result.get("timeshift_B", np.nan)
                ),
                "fit_metric": _finite_float(
                    result.get("fit_metric", np.nan)
                ),
                "n_result_A": result.get("n_A", np.nan),
                "n_result_B": result.get("n_B", np.nan),
                "n_result_valid_A": result.get("n_valid_A", np.nan),
                "n_result_valid_B": result.get("n_valid_B", np.nan),
                "warning_count": result.get("warning_count", 0),
                "failure_stage": result.get("failure_stage"),
                "exception_type": result.get("exception_type"),
                "exception_message": result.get("exception_message"),
                "runtime_seconds": _finite_float(
                    result.get("runtime_seconds", np.nan)
                ),
                "model_gate_accurate": model_gate_accurate,
                "coords": source_row.get("coords"),
            }

            accurate_bool = (
                False if accurate is pd.NA else bool(accurate)
            )
            row["usable"] = row["assessable"] and accurate_bool
            rows.append(row)

    table = pd.DataFrame(rows, columns=[
        "system_id", "band", "curve_type", "estimator", "true_delay", "status",
        "attempted", "converged", "assessable", "accurate", "estimated_delay",
        "delay_error", "abs_delay_error", "relative_delay_error", "accuracy_tolerance",
        "timeshift_A", "timeshift_B", "fit_metric", "n_result_A", "n_result_B",
        "n_result_valid_A", "n_result_valid_B", "warning_count", "failure_stage",
        "exception_type", "exception_message", "runtime_seconds", "model_gate_accurate",
        "coords", "usable",
    ])
    if not table.empty:
        table["accurate"] = pd.array(table["accurate"], dtype="boolean")
        table["model_gate_accurate"] = pd.array(
            table["model_gate_accurate"], dtype="boolean"
        )
        for column in ("attempted", "converged", "assessable", "usable"):
            table[column] = table[column].fillna(False).astype(bool)
    return table


def add_population_derived_columns(
    analysis_table: pd.DataFrame,
) -> pd.DataFrame:
    """Add stable plotting columns without assuming they already exist."""
    table = analysis_table.copy()

    table["available"] = table["feature_status"].eq("success")
    table["accurate"] = pd.array(table["accurate"], dtype="boolean")
    table["usable"] = (
        table["assessable"].fillna(False).astype(bool)
        & table["accurate"].fillna(False).astype(bool)
    )

    numeric_columns = [
        "n_min",
        "overlap_baseline",
        "median_cadence",
        "max_gap",
        "median_error",
        "true_delay",
        "abs_delay_error",
    ]
    for column in numeric_columns:
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce")
        else:
            table[column] = np.nan

    table["log10_n_min_plus1"] = np.log10(
        table["n_min"].clip(lower=0.0) + 1.0
    )
    table["log10_overlap_plus1"] = np.log10(
        table["overlap_baseline"].clip(lower=0.0) + 1.0
    )
    table["log10_median_cadence"] = np.log10(
        table["median_cadence"].clip(lower=1e-3)
    )
    table["log10_max_gap_plus1"] = np.log10(
        table["max_gap"].clip(lower=0.0) + 1.0
    )
    table["log10_median_error"] = np.log10(
        table["median_error"].clip(lower=1e-6)
    )
    table["log10_abs_true_delay_plus1"] = np.log10(
        np.abs(table["true_delay"]) + 1.0
    )
    table["log10_abs_delay_error_plus"] = np.log10(
        table["abs_delay_error"].clip(lower=1e-3)
    )

    return table


def build_population_analysis_tables(
    lens_systems: Mapping,
    baseline_results_by_estimator: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, pd.DataFrame]:
    """Build feature, result, merged analysis, and band-summary tables."""
    result_tables = [
        flatten_baseline_results(results, estimator=estimator)
        for estimator, results in baseline_results_by_estimator.items()
    ]
    results_table = (
        pd.concat(result_tables, ignore_index=True)
        if result_tables
        else flatten_baseline_results([], estimator="")
    )

    feature_table = build_light_curve_feature_table_from_lcs(
        lens_systems
    )

    analysis_table = results_table.merge(
        feature_table,
        on=["system_id", "band", "curve_type"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_feature"),
    )
    analysis_table = add_population_derived_columns(analysis_table)
    band_summary = summarize_band_performance(analysis_table)

    return {
        "results": results_table,
        "features": feature_table,
        "analysis": analysis_table,
        "band_summary": band_summary,
    }


def summarize_band_performance(
    analysis_table: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize availability, assessability, and accuracy by band."""
    rows: list[dict[str, Any]] = []

    group_columns = ["curve_type", "estimator", "band"]
    summary_columns = group_columns + [
        "n_rows", "n_available", "n_attempted", "n_assessable", "n_accurate",
        "attempted_fraction_given_available", "assessable_fraction_given_attempted",
        "accuracy_fraction_given_assessable", "end_to_end_accurate_fraction",
        "median_abs_delay_error", "rmse_delay",
    ]
    if analysis_table.empty:
        return pd.DataFrame(columns=summary_columns)
    for keys, group in analysis_table.groupby(group_columns, dropna=False):
        curve_type, estimator, band = keys

        available = group["available"].fillna(False).astype(bool)
        attempted = group["attempted"].fillna(False).astype(bool)
        assessable = group["assessable"].fillna(False).astype(bool)
        accurate = group["accurate"].fillna(False).astype(bool)
        usable = assessable & accurate

        n_available = int(available.sum())
        n_attempted = int((available & attempted).sum())
        n_assessable = int((available & attempted & assessable).sum())
        n_accurate = int((available & attempted & assessable & accurate).sum())

        assessable_errors = pd.to_numeric(
            group.loc[available & attempted & assessable, "abs_delay_error"],
            errors="coerce",
        )

        rows.append(
            {
                "curve_type": curve_type,
                "estimator": estimator,
                "band": band,
                "n_rows": int(len(group)),
                "n_available": n_available,
                "n_attempted": n_attempted,
                "n_assessable": n_assessable,
                "n_accurate": n_accurate,
                "attempted_fraction_given_available": (
                    n_attempted / n_available
                    if n_available
                    else np.nan
                ),
                "assessable_fraction_given_attempted": (
                    n_assessable / n_attempted
                    if n_attempted
                    else np.nan
                ),
                "accuracy_fraction_given_assessable": (
                    n_accurate / n_assessable
                    if n_assessable
                    else np.nan
                ),
                "end_to_end_accurate_fraction": (
                    n_accurate / n_available
                    if n_available
                    else np.nan
                ),
                "median_abs_delay_error": (
                    float(assessable_errors.median())
                    if assessable_errors.notna().any()
                    else np.nan
                ),
                "rmse_delay": (
                    float(
                        np.sqrt(
                            np.nanmean(
                                np.square(
                                    pd.to_numeric(
                                        group.loc[
                                            available
                                            & attempted
                                            & assessable,
                                            "delay_error",
                                        ],
                                        errors="coerce",
                                    )
                                )
                            )
                        )
                    )
                    if n_assessable
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def binary_feature_correlations(
    analysis_table: pd.DataFrame,
    *,
    features: Sequence[str],
    outcome: str,
) -> pd.DataFrame:
    """Point-biserial correlations for exploratory diagnostics."""
    from scipy.stats import pointbiserialr

    rows = []
    for feature in features:
        data = (
            analysis_table[[feature, outcome]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(data) < 3 or data[outcome].nunique() < 2 or data[feature].nunique() < 2:
            continue
        coefficient, p_value = pointbiserialr(
            data[outcome].astype(int),
            data[feature].astype(float),
        )
        rows.append(
            {
                "feature": feature,
                "outcome": outcome,
                "n": len(data),
                "correlation": coefficient,
                "p_value_naive": p_value,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "outcome",
                "n",
                "correlation",
                "p_value_naive",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        "correlation", key=np.abs, ascending=False
    ).reset_index(drop=True)


def continuous_error_correlations(
    analysis_table: pd.DataFrame,
    *,
    features: Sequence[str],
    error_column: str = "abs_delay_error",
) -> pd.DataFrame:
    """Spearman correlations with a continuous delay-error measure."""
    from scipy.stats import spearmanr

    rows = []
    for feature in features:
        data = (
            analysis_table[[feature, error_column]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(data) < 3 or data[feature].nunique() < 2 or data[error_column].nunique() < 2:
            continue
        coefficient, p_value = spearmanr(
            data[feature].astype(float),
            data[error_column].astype(float),
        )
        rows.append(
            {
                "feature": feature,
                "error_column": error_column,
                "n": len(data),
                "spearman_r": coefficient,
                "p_value_naive": p_value,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "error_column",
                "n",
                "spearman_r",
                "p_value_naive",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        "spearman_r", key=np.abs, ascending=False
    ).reset_index(drop=True)


def classify_mock_measurement_success(
    mock_table: pd.DataFrame,
    *,
    minimum_finite_mocks: int = 400,
    maximum_failure_fraction: float = 0.20,
    maximum_catastrophic_fraction: float = 0.20,
    maximum_relative_uncertainty: float = 0.20,
    minimum_delay_significance: float = 3.0,
    require_baseline_accuracy: bool = True,
    require_truth_coverage: bool = True,
    truth_coverage_sigma: float = 2.0,
) -> pd.DataFrame:
    """Classify mock-calibrated time-delay measurements.

    Three flags are produced:

    ``mock_calibrated``
        The production mock run completed with enough finite realizations and
        acceptable optimizer/outlier fractions.

    ``constrained``
        A truth-independent, real-data-style criterion: the delay is finite,
        its calibrated uncertainty is finite and positive, relative precision
        is adequate, and the delay differs from zero by the requested number
        of sigma.

    ``validated_success``
        The simulation-validation criterion. It starts from ``constrained`` and
        can additionally require the notebook's predefined accuracy flag and
        consistency with the known truth within ``truth_coverage_sigma``.
    """
    table = mock_table.copy()

    numeric_columns = [
        "production_n_finite", "observed_mock_failure_fraction",
        "observed_mock_catastrophic_fraction", "observed_mock_total_error",
        "observed_delay_raw", "relative_uncertainty_measured", "delay_significance",
        "truth_residual_raw",
    ]
    numeric = table.reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce")
    n_finite = numeric["production_n_finite"]
    failure_fraction = numeric["observed_mock_failure_fraction"]
    catastrophic_fraction = numeric["observed_mock_catastrophic_fraction"]
    sigma = numeric["observed_mock_total_error"]
    delay = numeric["observed_delay_raw"]
    relative_uncertainty = numeric["relative_uncertainty_measured"]
    significance = numeric["delay_significance"]

    production_status = table.get(
        "production_status", pd.Series(index=table.index, dtype=object)
    )
    production_calibrated = table.get(
        "production_calibrated", pd.Series(False, index=table.index)
    ).fillna(False).astype(bool)

    table["mock_calibrated"] = (
        table.get("mock_status", pd.Series(index=table.index, dtype=object)).eq("success")
        & production_status.eq("success")
        & production_calibrated
        & n_finite.ge(minimum_finite_mocks)
        & failure_fraction.le(maximum_failure_fraction)
        & catastrophic_fraction.le(maximum_catastrophic_fraction)
    )

    table["measured_successfully"] = (
        table.get("assessable", pd.Series(False, index=table.index)).fillna(False).astype(bool)
        & np.isfinite(delay)
    )

    table["constrained"] = (
        table["mock_calibrated"]
        & table["measured_successfully"]
        & np.isfinite(sigma)
        & sigma.gt(0.0)
        & relative_uncertainty.le(maximum_relative_uncertainty)
        & significance.ge(minimum_delay_significance)
    )

    validated = table["constrained"].copy()

    if require_baseline_accuracy:
        accurate = table.get(
            "accurate", pd.Series(False, index=table.index)
        ).fillna(False).astype(bool)
        validated &= accurate

    if require_truth_coverage:
        residual = numeric["truth_residual_raw"]
        validated &= residual.abs().le(truth_coverage_sigma * sigma)

    table["validated_success"] = validated

    table["selection_reason"] = "not_selected"
    table.loc[~table["measured_successfully"], "selection_reason"] = (
        "observed_delay_not_assessable"
    )
    table.loc[
        table["measured_successfully"] & ~table["mock_calibrated"],
        "selection_reason",
    ] = "production_mock_calibration_failed"
    table.loc[
        table["mock_calibrated"]
        & relative_uncertainty.gt(maximum_relative_uncertainty),
        "selection_reason",
    ] = "relative_uncertainty_too_large"
    table.loc[
        table["mock_calibrated"]
        & significance.lt(minimum_delay_significance),
        "selection_reason",
    ] = "delay_not_significantly_nonzero"

    if require_baseline_accuracy and "accurate" in table:
        table.loc[
            table["constrained"]
            & ~table["accurate"].fillna(False).astype(bool),
            "selection_reason",
        ] = "inaccurate_against_simulation_truth"

    if require_truth_coverage:
        residual = numeric["truth_residual_raw"]
        table.loc[
            table["constrained"]
            & residual.abs().gt(truth_coverage_sigma * sigma),
            "selection_reason",
        ] = "truth_outside_calibrated_interval"

    table.loc[table["validated_success"], "selection_reason"] = "selected"

    table.attrs["selection_criteria"] = {
        "minimum_finite_mocks": int(minimum_finite_mocks),
        "maximum_failure_fraction": float(maximum_failure_fraction),
        "maximum_catastrophic_fraction": float(maximum_catastrophic_fraction),
        "maximum_relative_uncertainty": float(maximum_relative_uncertainty),
        "minimum_delay_significance": float(minimum_delay_significance),
        "require_baseline_accuracy": bool(require_baseline_accuracy),
        "require_truth_coverage": bool(require_truth_coverage),
        "truth_coverage_sigma": float(truth_coverage_sigma),
    }

    return table


def add_estimator_agreement(
    selection_table: pd.DataFrame,
    *,
    maximum_disagreement_sigma: float = 2.0,
) -> pd.DataFrame:
    """Add spline-versus-regdiff agreement within each system and band.

    The agreement statistic is

    ``abs(delay_spline - delay_regdiff) / sqrt(sigma_spline**2 + sigma_regdiff**2)``.

    Rows with only one calibrated estimator retain ``NaN`` agreement values.
    """
    table = selection_table.copy()
    if table.empty:
        table["estimator_disagreement_sigma"] = pd.Series(index=table.index, dtype=float)
        table["estimators_agree"] = pd.Series(index=table.index, dtype=bool)
        return table
    candidate = table.loc[
        table["constrained"],
        [
            "system_id",
            "band",
            "estimator",
            "observed_delay_raw",
            "observed_mock_total_error",
        ],
    ]

    delays = candidate.pivot_table(
        index=["system_id", "band"],
        columns="estimator",
        values="observed_delay_raw",
        aggfunc="first",
    )
    errors = candidate.pivot_table(
        index=["system_id", "band"],
        columns="estimator",
        values="observed_mock_total_error",
        aggfunc="first",
    )

    agreement = pd.DataFrame(index=delays.index)
    if {"spline", "regdiff"}.issubset(delays.columns) and {
        "spline",
        "regdiff",
    }.issubset(errors.columns):
        denominator = np.sqrt(errors["spline"] ** 2 + errors["regdiff"] ** 2)
        agreement["estimator_disagreement_sigma"] = (
            (delays["spline"] - delays["regdiff"]).abs()
            / denominator.replace(0.0, np.nan)
        )
        agreement["estimators_agree"] = agreement[
            "estimator_disagreement_sigma"
        ].le(maximum_disagreement_sigma)
    else:
        agreement["estimator_disagreement_sigma"] = np.nan
        agreement["estimators_agree"] = False

    agreement = agreement.reset_index()
    return table.merge(
        agreement,
        on=["system_id", "band"],
        how="left",
        validate="many_to_one",
    )


def _as_float(value: Any, default: float = np.nan) -> float:
    """Convert a scalar-like value to float without raising."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if np.isfinite(converted) else default


def _flatten_mock_row(row: Mapping[str, Any], estimator: str) -> dict[str, Any]:
    """Flatten one row from ``spline_mock_results`` or ``regdiff_mock_results``."""
    diagnostic = row.get("diagnostic_result") or {}
    production = row.get("production_result") or {}

    flat: dict[str, Any] = {
        "system_id": row.get("system_id"),
        "band": row.get("band"),
        "estimator": row.get("estimator", estimator),
        "mock_status": row.get("mock_status", "not_run"),
        "observed_delay_raw": _as_float(row.get("observed_delay_raw")),
        "observed_delay_bias_corrected": _as_float(
            row.get("observed_delay_bias_corrected")
        ),
        "observed_mock_bias": _as_float(row.get("observed_mock_bias")),
        "observed_mock_random_error": _as_float(
            row.get("observed_mock_random_error")
        ),
        "observed_mock_total_error": _as_float(
            row.get("observed_mock_total_error")
        ),
        "observed_mock_failure_fraction": _as_float(
            row.get("observed_mock_failure_fraction")
        ),
        "observed_mock_catastrophic_fraction": _as_float(
            row.get("observed_mock_catastrophic_fraction")
        ),
    }

    diagnostic_keys = (
        "status",
        "mockable",
        "calibrated",
        "n_requested",
        "n_returned",
        "n_finite",
        "failure_fraction",
        "median_bias",
        "random_error",
        "total_error",
        "rmse",
        "catastrophic_fraction",
        "runtime_seconds",
        "warning_count",
    )
    production_keys = diagnostic_keys + (
        "mean_bias",
        "error_p16",
        "error_p50",
        "error_p84",
        "measured_delay_p16",
        "measured_delay_p50",
        "measured_delay_p84",
        "true_delay_p16",
        "true_delay_p50",
        "true_delay_p84",
        "fitted_delay",
    )

    for key in diagnostic_keys:
        flat[f"diagnostic_{key}"] = diagnostic.get(key, np.nan)

    for key in production_keys:
        flat[f"production_{key}"] = production.get(key, np.nan)

    # Prefer the flattened summary written by the notebook, but fall back to
    # values inside production_result when necessary.
    fallback_map = {
        "observed_mock_bias": "median_bias",
        "observed_mock_random_error": "random_error",
        "observed_mock_total_error": "total_error",
        "observed_mock_failure_fraction": "failure_fraction",
        "observed_mock_catastrophic_fraction": "catastrophic_fraction",
    }
    for output_key, production_key in fallback_map.items():
        if not np.isfinite(_as_float(flat[output_key])):
            flat[output_key] = _as_float(production.get(production_key))

    return flat


def build_mock_measurement_table(
    analysis_table: pd.DataFrame,
    mock_results_by_estimator: Mapping[str, Sequence[Mapping[str, Any]]],
) -> pd.DataFrame:
    """Build one mock-calibrated row per system, band, and estimator.

    Parameters
    ----------
    analysis_table
        The ``analysis_df`` produced by ``build_population_analysis_tables``.
    mock_results_by_estimator
        Mapping such as::

            {
                "spline": spline_mock_results,
                "regdiff": regdiff_mock_results,
            }

    Returns
    -------
    pandas.DataFrame
        A table containing the observed point estimate, known simulation truth,
        production-mock uncertainty diagnostics, precision, significance, and
        confidence-interval coverage.
    """
    rows: list[dict[str, Any]] = []
    for estimator, mock_rows in mock_results_by_estimator.items():
        rows.extend(_flatten_mock_row(row, estimator) for row in mock_rows)

    mock_table = pd.DataFrame(rows)
    if mock_table.empty:
        return mock_table

    observed = analysis_table.loc[
        analysis_table["curve_type"].eq("observed")
    ].copy()

    preferred_columns = [
        "system_id",
        "band",
        "estimator",
        "available",
        "attempted",
        "assessable",
        "accurate",
        "usable",
        "true_delay",
        "estimated_delay",
        "delay_error",
        "abs_delay_error",
        "relative_delay_error",
        "accuracy_tolerance",
        "feature_status",
        "n_A",
        "n_B",
        "n_min",
        "n_total",
        "n_shared",
        "shared_fraction",
        "overlap_baseline",
        "median_cadence",
        "max_gap",
        "median_error",
        "median_image_contrast",
        "delay_to_overlap_ratio",
        "z_lens",
        "z_source",
        "einstein_radius",
        "batch",
        "coords",
    ]
    merge_columns = [column for column in preferred_columns if column in observed]

    observed = observed[merge_columns].drop_duplicates(
        subset=["system_id", "band", "estimator"]
    )

    table = mock_table.merge(
        observed,
        on=["system_id", "band", "estimator"],
        how="left",
        validate="one_to_one",
    )

    # The mock loop records the same delay as the observed baseline result. Use
    # the baseline value as a fallback for rows that stopped before row.update().
    raw_delay = pd.to_numeric(table["observed_delay_raw"], errors="coerce")
    if "estimated_delay" in table:
        baseline_delay = pd.to_numeric(table["estimated_delay"], errors="coerce")
        table["observed_delay_raw"] = raw_delay.where(
            raw_delay.notna(), baseline_delay
        )

    table["observed_mock_total_error"] = pd.to_numeric(
        table["observed_mock_total_error"], errors="coerce"
    )
    table["observed_mock_random_error"] = pd.to_numeric(
        table["observed_mock_random_error"], errors="coerce"
    )
    table["observed_mock_bias"] = pd.to_numeric(
        table["observed_mock_bias"], errors="coerce"
    )

    finite_raw = np.isfinite(table["observed_delay_raw"])
    finite_bias = np.isfinite(table["observed_mock_bias"])
    table.loc[finite_raw & finite_bias, "observed_delay_bias_corrected"] = (
        table.loc[finite_raw & finite_bias, "observed_delay_raw"]
        - table.loc[finite_raw & finite_bias, "observed_mock_bias"]
    )

    sigma = table["observed_mock_total_error"]
    measured_abs = table["observed_delay_raw"].abs()
    true_delay = pd.to_numeric(
        table.get("true_delay", pd.Series(np.nan, index=table.index)), errors="coerce"
    )
    true_abs = true_delay.abs()

    table["relative_uncertainty_measured"] = sigma / measured_abs.replace(0.0, np.nan)
    table["relative_uncertainty_true"] = sigma / true_abs.replace(0.0, np.nan)
    table["delay_significance"] = measured_abs / sigma.replace(0.0, np.nan)

    table["truth_residual_raw"] = (
        table["observed_delay_raw"] - true_delay
    )
    table["truth_pull_raw"] = table["truth_residual_raw"] / sigma.replace(0.0, np.nan)

    for label, n_sigma in (("68", 1.0), ("95", 1.96)):
        table[f"ci{label}_low"] = table["observed_delay_raw"] - n_sigma * sigma
        table[f"ci{label}_high"] = table["observed_delay_raw"] + n_sigma * sigma
        table[f"truth_covered_{label}"] = (
            true_delay.ge(table[f"ci{label}_low"])
            & true_delay.le(table[f"ci{label}_high"])
        )

    return table


def save_successful_time_delay_systems(
    lens_systems: Mapping[Any, Mapping[str, Any]],
    selection_table: pd.DataFrame,
    output_directory: str | Path,
    *,
    selection_column: str = "validated_success",
    file_stem: str = "successful_time_delay",
    save_per_band: bool = True,
    save_per_estimator: bool = True,
) -> dict[str, Path]:
    """Save selected measurements and their complete system dictionaries.

    The full-system pickle preserves the objects stored in ``lens_systems`` and
    adds a top-level ``successful_time_delay_measurements`` list to each copied
    system dictionary. The input mapping is not modified.
    """
    if selection_column not in selection_table:
        raise KeyError(f"Missing selection column: {selection_column!r}")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    selected = selection_table.loc[
        selection_table[selection_column].fillna(False).astype(bool)
    ].copy()
    if selected.empty:
        for column in ("system_id", "band", "estimator"):
            if column not in selected:
                selected[column] = pd.Series(index=selected.index, dtype=object)
    selected = selected.sort_values(
        ["system_id", "band", "estimator"]
    ).reset_index(drop=True)

    system_ids = selected["system_id"].drop_duplicates().tolist()

    # Exclude bulky/debug columns from the records embedded in every system.
    embedded_columns = [
        column
        for column in selected.columns
        if column not in {"coords"}
    ]

    selected_systems: dict[Any, dict[str, Any]] = {}
    for system_id in system_ids:
        if system_id not in lens_systems:
            continue

        system_copy = dict(lens_systems[system_id])
        records = selected.loc[
            selected["system_id"].eq(system_id), embedded_columns
        ].to_dict(orient="records")
        system_copy["successful_time_delay_measurements"] = records
        selected_systems[system_id] = system_copy

    measurement_pickle = output_directory / f"{file_stem}_measurements.pkl"
    measurement_csv = output_directory / f"{file_stem}_measurements.csv"
    systems_pickle = output_directory / f"{file_stem}_systems.pkl"
    ids_pickle = output_directory / f"{file_stem}_system_ids.pkl"
    criteria_json = output_directory / f"{file_stem}_selection_criteria.json"

    print(f"Writing to {measurement_pickle}...")
    selected.to_pickle(measurement_pickle)
    print(f"Writing to {measurement_csv}...")
    selected.drop(columns=["coords"], errors="ignore").to_csv(
        measurement_csv, index=False
    )

    print(f"Writing to {systems_pickle}...")
    with systems_pickle.open("wb") as handle:
        pickle.dump(selected_systems, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Writing to {ids_pickle}...")
    with ids_pickle.open("wb") as handle:
        pickle.dump(system_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    criteria = dict(selection_table.attrs.get("selection_criteria", {}))
    criteria.update(
        {
            "selection_column": selection_column,
            "n_selected_measurements": int(len(selected)),
            "n_selected_systems": int(len(selected_systems)),
        }
    )
    print(f"Writing to {criteria_json}...")
    criteria_json.write_text(json.dumps(criteria, indent=2), encoding="utf-8")

    paths: dict[str, Path] = {
        "measurements_pickle": measurement_pickle,
        "measurements_csv": measurement_csv,
        "systems_pickle": systems_pickle,
        "system_ids_pickle": ids_pickle,
        "criteria_json": criteria_json,
    }

    if save_per_band:
        for band, group in selected.groupby("band", dropna=False):
            band_name = str(band)
            path = output_directory / f"{file_stem}_measurements_band_{band_name}.pkl"
            print(f"Writing to {path}...")
            group.to_pickle(path)
            paths[f"band_{band_name}"] = path

    if save_per_estimator:
        for estimator, group in selected.groupby("estimator", dropna=False):
            estimator_name = str(estimator)
            path = (
                output_directory
                / f"{file_stem}_measurements_estimator_{estimator_name}.pkl"
            )
            print(f"Writing to {path}...")
            group.to_pickle(path)
            paths[f"estimator_{estimator_name}"] = path

    return paths
