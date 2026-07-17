"""PyCS3 utilities for lensed-supernova time-delay experiments.

This module consolidates the notebook helpers used in the accompanying
analysis and adds two capabilities:

1. Truth-independent multistart optimization for the spline and
   regression-difference estimators.
2. CPU-parallel execution of ``pycs3.sim.run.multirun`` at the pickle-file
   level.

The parallel path uses the third-party ``multiprocess`` package because it
serializes scientific Python objects more reliably than the standard-library
``multiprocessing`` module in notebooks. Install it with::

    python -m pip install multiprocess threadpoolctl

GPU execution is not implemented because the PyCS3 spline and regression-
difference optimizers do not expose a CUDA backend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Any
import json
import os
import pickle
import random
import shutil
import traceback
import warnings

import numpy as np
import pandas as pd

import pycs3
import pycs3.gen.lc_func
import pycs3.gen.mrg
import pycs3.gen.stat
import pycs3.regdiff.multiopt
import pycs3.sim.draw
import pycs3.sim.run
import pycs3.spl.topopt

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - optional runtime dependency
    threadpool_limits = None


# ---------------------------------------------------------------------------
# Light-curve construction
# ---------------------------------------------------------------------------


def make_pycs3_curve_sets(
    times: np.ndarray,
    model_magnitudes: np.ndarray,
    observed_magnitudes: np.ndarray,
    magnitude_errors: np.ndarray,
    *,
    telescope_name: str = "LSST",
    minimum_points: int = 2,
) -> dict[str, Any]:
    """Construct matched ideal-model and observed PyCS3 curve pairs.

    A separate finite-data mask is constructed for each image, but the model
    and observed version of a given image use the same detected epochs. Sparse
    curves are retained as long as they satisfy the structural minimum needed
    to attempt a two-curve delay fit.
    """

    times = np.asarray(times, dtype=float)
    model_magnitudes = np.asarray(model_magnitudes, dtype=float)
    observed_magnitudes = np.asarray(observed_magnitudes, dtype=float)
    magnitude_errors = np.asarray(magnitude_errors, dtype=float)

    expected_shape = (len(times), 2)
    for name, array in {
        "model_magnitudes": model_magnitudes,
        "observed_magnitudes": observed_magnitudes,
        "magnitude_errors": magnitude_errors,
    }.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {array.shape}; expected {expected_shape}."
            )

    ideal_model_lcs: list[Any] = []
    observed_lcs: list[Any] = []
    diagnostics: dict[str, Any] = {}

    for image_index, image_name in enumerate(("A", "B")):
        model_mag = model_magnitudes[:, image_index]
        observed_mag = observed_magnitudes[:, image_index]
        mag_error = magnitude_errors[:, image_index]

        valid = (
            np.isfinite(times)
            & np.isfinite(model_mag)
            & np.isfinite(observed_mag)
            & np.isfinite(mag_error)
            & (mag_error > 0.0)
        )

        valid_times = times[valid]
        valid_model_mag = model_mag[valid]
        valid_observed_mag = observed_mag[valid]
        valid_mag_error = mag_error[valid]

        order = np.argsort(valid_times)
        valid_times = valid_times[order]
        valid_model_mag = valid_model_mag[order]
        valid_observed_mag = valid_observed_mag[order]
        valid_mag_error = valid_mag_error[order]

        if len(valid_times) < int(minimum_points):
            raise ValueError(
                f"Image {image_name} has only {len(valid_times)} valid points; "
                f"at least {minimum_points} are required to construct the curve."
            )

        ideal_model_lcs.append(
            pycs3.gen.lc_func.factory(
                valid_times,
                valid_model_mag,
                valid_mag_error,
                telescopename=telescope_name,
                object=image_name,
                verbose=False,
            )
        )
        observed_lcs.append(
            pycs3.gen.lc_func.factory(
                valid_times,
                valid_observed_mag,
                valid_mag_error,
                telescopename=telescope_name,
                object=image_name,
                verbose=False,
            )
        )

        diagnostics[f"n_valid_{image_name}"] = int(len(valid_times))
        diagnostics[f"n_removed_{image_name}"] = int(len(times) - len(valid_times))
        diagnostics[f"median_error_{image_name}"] = float(
            np.median(valid_mag_error)
        )
        if len(valid_times) >= 2:
            diagnostics[f"baseline_{image_name}"] = float(
                valid_times[-1] - valid_times[0]
            )
            diagnostics[f"median_cadence_{image_name}"] = float(
                np.median(np.diff(valid_times))
            )
        else:
            diagnostics[f"baseline_{image_name}"] = np.nan
            diagnostics[f"median_cadence_{image_name}"] = np.nan

    pycs3.gen.mrg.colourise(ideal_model_lcs)
    pycs3.gen.mrg.colourise(observed_lcs)

    return {
        "ideal_model_lcs": ideal_model_lcs,
        "observed_lcs": observed_lcs,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Importable base optimizers
# ---------------------------------------------------------------------------


def spline_optimizer(
    lcs,
    kn: float = 20.0,
    *,
    rough_nit: int = 5,
    fine_nit: int = 15,
    rough_knotstep: float | None = None,
):
    """Run the notebook's rough + fine free-knot spline optimizer."""

    kn = float(kn)
    rough_step = (
        max(30.0, kn) if rough_knotstep is None else float(rough_knotstep)
    )
    pycs3.spl.topopt.opt_rough(
        lcs,
        nit=int(rough_nit),
        knotstep=rough_step,
        verbose=False,
    )
    return pycs3.spl.topopt.opt_fine(
        lcs,
        nit=int(fine_nit),
        knotstep=kn,
        verbose=False,
    )


def regdiff_optimizer(
    lcs,
    *,
    pd: int = 2,
    covkernel: str = "matern",
    pow: float = 1.5,
    amp: float = 1.0,
    scale: float = 200.0,
    errscale: float = 1.0,
    method: str = "weights",
):
    """Run the regression-difference time-shift optimizer."""

    return pycs3.regdiff.multiopt.opt_ts(
        lcs,
        pd=int(pd),
        covkernel=str(covkernel),
        pow=float(pow),
        amp=float(amp),
        scale=float(scale),
        errscale=float(errscale),
        verbose=False,
        method=str(method),
    )


# ---------------------------------------------------------------------------
# Shared fit helpers
# ---------------------------------------------------------------------------


def _validate_dual_lcs(lcs) -> dict[str, Any]:
    if not isinstance(lcs, (list, tuple)):
        raise TypeError("lcs must be a list or tuple of LightCurve objects.")
    if len(lcs) != 2:
        raise ValueError(f"Expected two curves [A, B]; received {len(lcs)}.")

    diagnostics: dict[str, Any] = {}
    for index, label in enumerate(("A", "B")):
        lc = lcs[index]
        times = np.asarray(lc.jds, dtype=float)
        mags = np.asarray(lc.mags, dtype=float)
        errors = np.asarray(lc.magerrs, dtype=float)
        if not (len(times) == len(mags) == len(errors)):
            raise ValueError(f"Curve {label} has inconsistent array lengths.")
        valid = (
            np.isfinite(times)
            & np.isfinite(mags)
            & np.isfinite(errors)
            & (errors > 0.0)
        )
        diagnostics[f"n_{label}"] = int(len(times))
        diagnostics[f"n_valid_{label}"] = int(np.sum(valid))
        diagnostics[f"object_{label}"] = str(getattr(lc, "object", label))
    return diagnostics


def _apply_initial_shifts(
    lcs,
    *,
    timeshifts: Sequence[float] = (0.0, 0.0),
    magshifts: Sequence[float] = (0.0, 0.0),
) -> None:
    if len(timeshifts) != 2 or len(magshifts) != 2:
        raise ValueError("timeshifts and magshifts must each contain two values.")
    for lc in lcs:
        lc.resetshifts()
    pycs3.gen.lc_func.applyshifts(
        lcs,
        timeshifts=[float(timeshifts[0]), float(timeshifts[1])],
        magshifts=[float(magshifts[0]), float(magshifts[1])],
    )


def _extract_fit_metric(optimizer_output: Any) -> float:
    """Extract a lower-is-better objective when the optimizer exposes one."""

    if optimizer_output is None:
        return np.nan

    for attribute in (
        "lastr2nostab",
        "lastr2",
        "r2",
        "chi2",
        "objective",
        "score",
        "dispersion",
        "d2",
    ):
        if hasattr(optimizer_output, attribute):
            try:
                value = float(getattr(optimizer_output, attribute))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value

    if isinstance(optimizer_output, Mapping):
        for key in (
            "lastr2nostab",
            "r2",
            "chi2",
            "objective",
            "score",
            "dispersion",
            "d2",
        ):
            try:
                value = float(optimizer_output[key])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value

    if isinstance(optimizer_output, tuple):
        for item in optimizer_output[1:]:
            if np.isscalar(item):
                try:
                    value = float(item)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    return value

    return np.nan


def _capture_curve_state(lcs) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for lc in lcs:
        state: dict[str, Any] = {}
        for attribute in ("timeshift", "magshift", "fluxshift", "ml"):
            if hasattr(lc, attribute):
                state[attribute] = deepcopy(getattr(lc, attribute))
        states.append(state)
    return states


def _restore_curve_state(lcs, states: Sequence[Mapping[str, Any]]) -> None:
    if len(lcs) != len(states):
        raise ValueError("Curve state count does not match the light-curve count.")
    for lc, state in zip(lcs, states):
        for attribute, value in state.items():
            setattr(lc, attribute, deepcopy(value))


def _delay_from_lcs(lcs) -> float:
    return float(lcs[1].timeshift) - float(lcs[0].timeshift)


def _starting_delay_grid(
    *,
    maximum_shift: float,
    number_of_starts: int,
    starting_delays: Sequence[float] | None,
) -> np.ndarray:
    if starting_delays is not None:
        grid = np.asarray(starting_delays, dtype=float).ravel()
    else:
        if number_of_starts < 1:
            raise ValueError("number_of_starts must be at least one.")
        maximum_shift = float(maximum_shift)
        if maximum_shift < 0.0:
            raise ValueError("maximum_shift cannot be negative.")
        grid = np.linspace(-maximum_shift, maximum_shift, int(number_of_starts))

    grid = grid[np.isfinite(grid)]
    if grid.size == 0:
        raise ValueError("No finite starting delays were supplied.")
    return np.unique(grid)


def _cluster_candidate_indices(
    delays: np.ndarray,
    tolerance: float,
) -> list[np.ndarray]:
    """Cluster one-dimensional delay candidates by adjacent separation."""

    if len(delays) == 0:
        return []
    order = np.argsort(delays)
    clusters: list[list[int]] = [[int(order[0])]]
    for previous, current in zip(order[:-1], order[1:]):
        if abs(float(delays[current]) - float(delays[previous])) <= tolerance:
            clusters[-1].append(int(current))
        else:
            clusters.append([int(current)])
    return [np.asarray(cluster, dtype=int) for cluster in clusters]


def _select_multistart_index(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selection_mode: str,
    consensus_tolerance_days: float,
) -> tuple[int, str]:
    delays = np.asarray([candidate["delay"] for candidate in candidates], dtype=float)
    metrics = np.asarray([candidate["fit_metric"] for candidate in candidates], dtype=float)
    finite_metrics = np.isfinite(metrics)

    mode = str(selection_mode).lower()
    if mode not in {"metric", "consensus", "hybrid"}:
        raise ValueError(
            "selection_mode must be 'metric', 'consensus', or 'hybrid'."
        )

    if mode == "metric" and np.any(finite_metrics):
        valid_indices = np.flatnonzero(finite_metrics)
        chosen = valid_indices[np.argmin(metrics[finite_metrics])]
        return int(chosen), "metric"

    clusters = _cluster_candidate_indices(
        delays,
        tolerance=float(consensus_tolerance_days),
    )
    if not clusters:
        raise RuntimeError("No converged multistart candidates are available.")

    def cluster_rank(indices: np.ndarray) -> tuple[float, float, float]:
        cluster_metrics = metrics[indices]
        finite = cluster_metrics[np.isfinite(cluster_metrics)]
        median_metric = float(np.median(finite)) if finite.size else np.inf
        spread = float(np.median(np.abs(delays[indices] - np.median(delays[indices]))))
        return (-float(len(indices)), median_metric, spread)

    winning_cluster = min(clusters, key=cluster_rank)

    if mode == "hybrid" and np.any(np.isfinite(metrics[winning_cluster])):
        cluster_finite = winning_cluster[np.isfinite(metrics[winning_cluster])]
        chosen = cluster_finite[np.argmin(metrics[cluster_finite])]
        return int(chosen), "hybrid"

    cluster_median = float(np.median(delays[winning_cluster]))
    distance = np.abs(delays[winning_cluster] - cluster_median)
    nearest = winning_cluster[np.flatnonzero(distance == np.min(distance))]
    if len(nearest) > 1 and np.any(np.isfinite(metrics[nearest])):
        finite_nearest = nearest[np.isfinite(metrics[nearest])]
        chosen = finite_nearest[np.argmin(metrics[finite_nearest])]
    else:
        chosen = nearest[0]

    fallback = "consensus" if mode != "metric" else "consensus_metric_unavailable"
    return int(chosen), fallback


@dataclass
class _MultistartOutcome:
    optimizer_output: Any
    candidates: list[dict[str, Any]]
    selected_index: int
    selected_start_delay: float
    selected_candidate_delay: float
    selected_candidate_metric: float
    final_delay: float
    final_metric: float
    selection_used: str
    warnings: list[str]


def _run_multistart_optimizer(
    lcs,
    *,
    optimizer: Callable,
    optimizer_kwargs: Mapping[str, Any] | None,
    maximum_shift: float = 60.0,
    number_of_starts: int = 13,
    starting_delays: Sequence[float] | None = None,
    selection_mode: str = "metric",
    consensus_tolerance_days: float = 2.0,
    initial_magshifts: Sequence[float] = (0.0, 0.0),
    refit_selected: bool = True,
) -> _MultistartOutcome:
    """Run one optimizer from a grid of initial B-minus-A shifts."""

    _validate_dual_lcs(lcs)
    optimizer_kwargs = dict(optimizer_kwargs or {})
    grid = _starting_delay_grid(
        maximum_shift=maximum_shift,
        number_of_starts=number_of_starts,
        starting_delays=starting_delays,
    )

    pristine_lcs = deepcopy(list(lcs))
    candidates: list[dict[str, Any]] = []
    warning_messages: list[str] = []

    for start_delay in grid:
        candidate_lcs = deepcopy(pristine_lcs)
        _apply_initial_shifts(
            candidate_lcs,
            timeshifts=(0.0, float(start_delay)),
            magshifts=initial_magshifts,
        )

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                output = optimizer(candidate_lcs, **optimizer_kwargs)
            warning_messages.extend(
                f"{item.category.__name__}: {item.message}" for item in caught
            )
            delay = _delay_from_lcs(candidate_lcs)
            shifts = [float(curve.timeshift) for curve in candidate_lcs]
            metric = _extract_fit_metric(output)
            if not np.all(np.isfinite([delay, *shifts])):
                continue
            candidates.append(
                {
                    "start_delay": float(start_delay),
                    "delay": float(delay),
                    "fit_metric": float(metric),
                    "curve_state": _capture_curve_state(candidate_lcs),
                }
            )
        except Exception as exc:
            candidates.append(
                {
                    "start_delay": float(start_delay),
                    "delay": np.nan,
                    "fit_metric": np.nan,
                    "curve_state": None,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )

    converged = [
        candidate
        for candidate in candidates
        if np.isfinite(candidate.get("delay", np.nan))
        and candidate.get("curve_state") is not None
    ]
    if not converged:
        raise RuntimeError("No multistart candidate returned a finite delay.")

    selected_index, selection_used = _select_multistart_index(
        converged,
        selection_mode=selection_mode,
        consensus_tolerance_days=consensus_tolerance_days,
    )
    selected = converged[selected_index]
    _restore_curve_state(lcs, selected["curve_state"])

    if refit_selected:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            final_output = optimizer(lcs, **optimizer_kwargs)
        warning_messages.extend(
            f"{item.category.__name__}: {item.message}" for item in caught
        )
    else:
        # The candidate optimizer output is intentionally not retained for all
        # starts. Re-running on the selected state keeps the returned model
        # consistent with the original ``lcs`` objects.
        final_output = optimizer(lcs, **optimizer_kwargs)

    final_delay = _delay_from_lcs(lcs)
    final_metric = _extract_fit_metric(final_output)
    if not np.isfinite(final_delay):
        raise RuntimeError("The selected multistart refit returned a nonfinite delay.")

    return _MultistartOutcome(
        optimizer_output=final_output,
        candidates=candidates,
        selected_index=int(selected_index),
        selected_start_delay=float(selected["start_delay"]),
        selected_candidate_delay=float(selected["delay"]),
        selected_candidate_metric=float(selected["fit_metric"]),
        final_delay=float(final_delay),
        final_metric=float(final_metric),
        selection_used=selection_used,
        warnings=warning_messages,
    )


# ---------------------------------------------------------------------------
# Safe baseline estimators
# ---------------------------------------------------------------------------


def safe_run_pycs3_dual(
    lcs,
    *,
    optimizer: Callable,
    estimator_name: str,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    initial_timeshifts: Sequence[float] = (0.0, 0.0),
    initial_magshifts: Sequence[float] = (0.0, 0.0),
    keep_objects: bool = False,
) -> dict[str, Any]:
    """Run one single-start estimator and always return a structured record."""

    start = perf_counter()
    result: dict[str, Any] = {
        "estimator": str(estimator_name),
        "optimizer_mode": "single_start",
        "status": "not_started",
        "converged": False,
        "delay": np.nan,
        "timeshift_A": np.nan,
        "timeshift_B": np.nan,
        "fit_metric": np.nan,
        "warning_count": 0,
        "warnings": [],
        "failure_stage": None,
        "exception_type": None,
        "exception_message": None,
        "traceback": None,
        "runtime_seconds": np.nan,
    }
    caught_warnings: list[Any] = []
    stage = "validation"

    try:
        result.update(_validate_dual_lcs(lcs))
        if result["n_valid_A"] < 2 or result["n_valid_B"] < 2:
            result.update(status="insufficient_finite_points", failure_stage=stage)
            return result

        stage = "copying_light_curves"
        work_lcs = deepcopy(list(lcs))
        stage = "initialization"
        _apply_initial_shifts(
            work_lcs,
            timeshifts=initial_timeshifts,
            magshifts=initial_magshifts,
        )
        stage = "optimization"
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            optimizer_output = optimizer(
                work_lcs,
                **dict(optimizer_kwargs or {}),
            )

        stage = "delay_extraction"
        result.update(
            timeshift_A=float(work_lcs[0].timeshift),
            timeshift_B=float(work_lcs[1].timeshift),
            delay=_delay_from_lcs(work_lcs),
            fit_metric=_extract_fit_metric(optimizer_output),
        )
        if not np.all(
            np.isfinite(
                [result["timeshift_A"], result["timeshift_B"], result["delay"]]
            )
        ):
            result.update(status="nonfinite_solution", failure_stage=stage)
            return result

        result.update(status="success", converged=True)
        if keep_objects:
            result["optimized_lcs"] = work_lcs
            result["optimizer_output"] = optimizer_output

    except Exception as exc:
        result.update(
            status=f"{stage}_failed",
            failure_stage=stage,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback=traceback.format_exc(limit=10),
        )
    finally:
        result["warnings"] = [
            f"{item.category.__name__}: {item.message}" for item in caught_warnings
        ]
        result["warning_count"] = len(result["warnings"])
        result["runtime_seconds"] = perf_counter() - start

    return result


def safe_run_pycs3_dual_multistart(
    lcs,
    *,
    optimizer: Callable,
    estimator_name: str,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    maximum_shift: float = 60.0,
    number_of_starts: int = 13,
    starting_delays: Sequence[float] | None = None,
    selection_mode: str = "metric",
    consensus_tolerance_days: float = 2.0,
    initial_magshifts: Sequence[float] = (0.0, 0.0),
    refit_selected: bool = True,
    keep_objects: bool = False,
) -> dict[str, Any]:
    """Run a dual-curve estimator from several initial time shifts.

    The truth is never used to choose a solution. Spline fits should normally
    use ``selection_mode='metric'``. Regression-difference fits can use
    ``selection_mode='consensus'`` when their return object does not expose a
    comparable scalar objective.
    """

    start = perf_counter()
    result: dict[str, Any] = {
        "estimator": str(estimator_name),
        "optimizer_mode": "multistart",
        "status": "not_started",
        "converged": False,
        "delay": np.nan,
        "timeshift_A": np.nan,
        "timeshift_B": np.nan,
        "fit_metric": np.nan,
        "maximum_shift": float(maximum_shift),
        "number_of_starts_requested": int(number_of_starts),
        "number_of_starts_converged": 0,
        "selected_start_delay": np.nan,
        "selected_candidate_delay": np.nan,
        "selection_mode_requested": str(selection_mode),
        "selection_mode_used": None,
        "candidate_start_delays": [],
        "candidate_delays": [],
        "candidate_fit_metrics": [],
        "candidate_statuses": [],
        "warning_count": 0,
        "warnings": [],
        "failure_stage": None,
        "exception_type": None,
        "exception_message": None,
        "traceback": None,
        "runtime_seconds": np.nan,
    }
    stage = "validation"

    try:
        result.update(_validate_dual_lcs(lcs))
        if result["n_valid_A"] < 2 or result["n_valid_B"] < 2:
            result.update(status="insufficient_finite_points", failure_stage=stage)
            return result

        stage = "copying_light_curves"
        work_lcs = deepcopy(list(lcs))
        stage = "multistart_optimization"
        outcome = _run_multistart_optimizer(
            work_lcs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            maximum_shift=maximum_shift,
            number_of_starts=number_of_starts,
            starting_delays=starting_delays,
            selection_mode=selection_mode,
            consensus_tolerance_days=consensus_tolerance_days,
            initial_magshifts=initial_magshifts,
            refit_selected=refit_selected,
        )

        result.update(
            status="success",
            converged=True,
            timeshift_A=float(work_lcs[0].timeshift),
            timeshift_B=float(work_lcs[1].timeshift),
            delay=float(outcome.final_delay),
            fit_metric=float(outcome.final_metric),
            number_of_starts_requested=len(outcome.candidates),
            number_of_starts_converged=sum(
                np.isfinite(candidate.get("delay", np.nan))
                for candidate in outcome.candidates
            ),
            selected_start_delay=outcome.selected_start_delay,
            selected_candidate_delay=outcome.selected_candidate_delay,
            selection_mode_used=outcome.selection_used,
            candidate_start_delays=[
                candidate.get("start_delay", np.nan)
                for candidate in outcome.candidates
            ],
            candidate_delays=[
                candidate.get("delay", np.nan) for candidate in outcome.candidates
            ],
            candidate_fit_metrics=[
                candidate.get("fit_metric", np.nan)
                for candidate in outcome.candidates
            ],
            candidate_statuses=[
                "success"
                if np.isfinite(candidate.get("delay", np.nan))
                else candidate.get("exception_type", "failed")
                for candidate in outcome.candidates
            ],
            warnings=outcome.warnings,
            warning_count=len(outcome.warnings),
        )
        if keep_objects:
            result["optimized_lcs"] = work_lcs
            result["optimizer_output"] = outcome.optimizer_output

    except Exception as exc:
        result.update(
            status=f"{stage}_failed",
            failure_stage=stage,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback=traceback.format_exc(limit=10),
        )
    finally:
        result["runtime_seconds"] = perf_counter() - start

    return result


def assess_delay_recovery(
    run_result: Mapping[str, Any],
    true_delay: float,
    *,
    absolute_tolerance: float = 1.0,
    relative_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Compare a returned delay with known simulation truth."""

    assessment = {
        "assessable": False,
        "true_delay": float(true_delay),
        "delay_error": np.nan,
        "absolute_delay_error": np.nan,
        "relative_delay_error": np.nan,
        "tolerance": np.nan,
        "accurate": False,
    }
    if not run_result.get("converged", False):
        return assessment

    estimated = float(run_result.get("delay", np.nan))
    truth = float(true_delay)
    if not np.all(np.isfinite([estimated, truth])):
        return assessment

    error = estimated - truth
    absolute_error = abs(error)
    tolerance = max(
        float(absolute_tolerance),
        float(relative_tolerance) * abs(truth),
    )
    assessment.update(
        assessable=True,
        delay_error=float(error),
        absolute_delay_error=float(absolute_error),
        relative_delay_error=(absolute_error / abs(truth) if truth != 0.0 else np.nan),
        tolerance=float(tolerance),
        accurate=bool(absolute_error <= tolerance),
    )
    return assessment


def pycs_signed_delay_from_arrival_times(
    arrival_time_delays: Sequence[float],
) -> float:
    """Return the truth matching ``timeshift_B - timeshift_A`` for [A, B].

    If B's feature arrives later, PyCS3 shifts B to earlier times, so the
    fitted B-minus-A shift has the opposite sign of ``arrival_B-arrival_A``.
    """

    values = np.asarray(arrival_time_delays, dtype=float).ravel()
    if values.size < 2:
        raise ValueError("At least two arrival-time delays are required.")
    return float(values[0] - values[1])


# ---------------------------------------------------------------------------
# Module-level multistart optimizer adapters used by PyCS3 multirun
# ---------------------------------------------------------------------------


def spline_multistart_optimizer(
    lcs,
    *,
    kn: float = 20.0,
    rough_nit: int = 5,
    fine_nit: int = 15,
    rough_knotstep: float | None = None,
    maximum_shift: float = 60.0,
    number_of_starts: int = 13,
    starting_delays: Sequence[float] | None = None,
    selection_mode: str = "metric",
    consensus_tolerance_days: float = 2.0,
    refit_selected: bool = True,
):
    """Multistart spline callable compatible with ``pycs3.sim.run.multirun``."""

    outcome = _run_multistart_optimizer(
        lcs,
        optimizer=spline_optimizer,
        optimizer_kwargs={
            "kn": kn,
            "rough_nit": rough_nit,
            "fine_nit": fine_nit,
            "rough_knotstep": rough_knotstep,
        },
        maximum_shift=maximum_shift,
        number_of_starts=number_of_starts,
        starting_delays=starting_delays,
        selection_mode=selection_mode,
        consensus_tolerance_days=consensus_tolerance_days,
        refit_selected=refit_selected,
    )
    return outcome.optimizer_output


def regdiff_multistart_optimizer(
    lcs,
    *,
    pd: int = 2,
    covkernel: str = "matern",
    pow: float = 1.5,
    amp: float = 1.0,
    scale: float = 200.0,
    errscale: float = 1.0,
    method: str = "weights",
    maximum_shift: float = 60.0,
    number_of_starts: int = 13,
    starting_delays: Sequence[float] | None = None,
    selection_mode: str = "consensus",
    consensus_tolerance_days: float = 2.0,
    refit_selected: bool = True,
):
    """Multistart regression-difference callable for ``multirun``.

    ``selection_mode='consensus'`` is the safe default because some PyCS3
    versions do not expose a scalar regression-difference objective in the
    return object. If your installed version exposes one, ``'metric'`` or
    ``'hybrid'`` can be used after validation.
    """

    outcome = _run_multistart_optimizer(
        lcs,
        optimizer=regdiff_optimizer,
        optimizer_kwargs={
            "pd": pd,
            "covkernel": covkernel,
            "pow": pow,
            "amp": amp,
            "scale": scale,
            "errscale": errscale,
            "method": method,
        },
        maximum_shift=maximum_shift,
        number_of_starts=number_of_starts,
        starting_delays=starting_delays,
        selection_mode=selection_mode,
        consensus_tolerance_days=consensus_tolerance_days,
        refit_selected=refit_selected,
    )
    return outcome.optimizer_output


# ---------------------------------------------------------------------------
# CPU-parallel multirun
# ---------------------------------------------------------------------------


def available_worker_count(
    *,
    requested: int | None = None,
    maximum: int | None = None,
) -> int:
    """Resolve a conservative worker count from Slurm or local CPU metadata."""

    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if requested is not None:
        allocated = min(allocated, int(requested))
    if maximum is not None:
        allocated = min(allocated, int(maximum))
    return max(1, allocated)


def _multirun_worker(arguments: tuple[Any, ...]) -> Any:
    (
        worker_id,
        simset,
        lcs,
        optimizer,
        optimizer_kwargs,
        optset,
        tsrand,
        keepopt,
        destpath,
        base_seed,
    ) = arguments

    seed = int(base_seed) + int(worker_id)
    np.random.seed(seed)
    random.seed(seed)
    sleep(0.15 * int(worker_id))

    limiter = threadpool_limits(limits=1) if threadpool_limits is not None else nullcontext()
    with limiter:
        return pycs3.sim.run.multirun(
            str(simset),
            lcs,
            optimizer,
            kwargs_optim=dict(optimizer_kwargs),
            optset=str(optset),
            tsrand=float(tsrand),
            keepopt=bool(keepopt),
            destpath=str(destpath),
        )


def run_multirun_parallel(
    *,
    simset: str,
    lcs,
    optimizer: Callable,
    optimizer_kwargs: Mapping[str, Any] | None,
    optset: str,
    tsrand: float,
    destpath: str | Path,
    nworkers: int = 1,
    max_work_units: int | None = None,
    keepopt: bool = False,
    base_seed: int = 93841,
) -> list[Any]:
    """Launch identical ``multirun`` workers on separate CPU processes.

    PyCS3's ``.workingon`` markers assign different simulation pickle files to
    each process. The optimizer must be defined at module scope; use the
    adapters above rather than a function defined inside a notebook cell.
    """

    optimizer_kwargs = dict(optimizer_kwargs or {})
    nworkers = max(1, int(nworkers))
    if max_work_units is not None:
        nworkers = min(nworkers, max(1, int(max_work_units)))

    if nworkers == 1:
        return [
            _multirun_worker(
                (
                    0,
                    simset,
                    lcs,
                    optimizer,
                    optimizer_kwargs,
                    optset,
                    tsrand,
                    keepopt,
                    Path(destpath),
                    base_seed,
                )
            )
        ]

    try:
        from multiprocess import Pool
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Parallel multirun requires the 'multiprocess' package. Install "
            "it with: python -m pip install multiprocess threadpoolctl"
        ) from exc

    jobs = [
        (
            worker_id,
            simset,
            lcs,
            optimizer,
            optimizer_kwargs,
            optset,
            tsrand,
            keepopt,
            Path(destpath),
            base_seed,
        )
        for worker_id in range(nworkers)
    ]
    with Pool(processes=nworkers) as pool:
        return pool.map(_multirun_worker, jobs)


# ---------------------------------------------------------------------------
# Mock generation and calibration
# ---------------------------------------------------------------------------


def run_pycs3_mocks(
    lcs,
    *,
    generative_optimizer: Callable,
    generative_optimizer_kwargs: Mapping[str, Any] | None,
    measurement_optimizer: Callable,
    measurement_optimizer_kwargs: Mapping[str, Any] | None,
    estimator_name: str,
    destpath: str | Path,
    simset: str,
    optset: str,
    n: int = 10,
    npkl: int = 5,
    nworkers: int = 1,
    base_seed: int = 93841,
    truetsr: float = 10.0,
    tsrand: float = 0.0,
    shotnoise: str | None = "magerrs",
    tweakml=None,
    tweakspl=None,
    initial_timeshifts: Sequence[float] = (0.0, 0.0),
    initial_magshifts: Sequence[float] = (0.0, 0.0),
    catastrophic_threshold_days: float | None = 5.0,
    catastrophic_threshold_percentage: float | None = 0.2,
    overwrite: bool = False,
    keep_runresults: bool = False,
    keep_fitted_objects: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Fit a generative model, draw mocks, and calibrate one estimator.

    Only the ``multirun`` optimization stage is parallelized. Mock drawing and
    result collection occur once in the parent process. When the measurement
    optimizer is itself multistart, keep ``tsrand=0`` so PyCS3 does not add a
    second, redundant random initialization outside the explicit start grid.
    """

    start_time = perf_counter()
    generative_optimizer_kwargs = dict(generative_optimizer_kwargs or {})
    measurement_optimizer_kwargs = dict(measurement_optimizer_kwargs or {})
    destpath = Path(destpath)
    simulation_directory = destpath / f"sims_{simset}"
    result_directory = destpath / f"sims_{simset}_opt_{optset}"
    requested_mocks = int(n) * int(npkl)

    result: dict[str, Any] = {
        "estimator": str(estimator_name),
        "simset": str(simset),
        "optset": str(optset),
        "status": "not_started",
        "mockable": False,
        "calibrated": False,
        "failure_stage": None,
        "n_requested": requested_mocks,
        "n_returned": 0,
        "n_finite": 0,
        "n_failed_or_missing": requested_mocks,
        "failure_fraction": 1.0,
        "median_bias": np.nan,
        "mean_bias": np.nan,
        "random_error": np.nan,
        "total_error": np.nan,
        "rmse": np.nan,
        "error_p16": np.nan,
        "error_p50": np.nan,
        "error_p84": np.nan,
        "measured_delay_p16": np.nan,
        "measured_delay_p50": np.nan,
        "measured_delay_p84": np.nan,
        "true_delay_p16": np.nan,
        "true_delay_p50": np.nan,
        "true_delay_p84": np.nan,
        "catastrophic_threshold_days": catastrophic_threshold_days,
        "catastrophic_threshold_percentage": catastrophic_threshold_percentage,
        "catastrophic_fraction": np.nan,
        "n_valid_A": 0,
        "n_valid_B": 0,
        "fitted_timeshift_A": np.nan,
        "fitted_timeshift_B": np.nan,
        "fitted_delay": np.nan,
        "nworkers_requested": int(nworkers),
        "nworkers_used": 0,
        "warning_count": 0,
        "warnings": [],
        "exception_type": None,
        "exception_message": None,
        "traceback": None,
        "simulation_directory": str(simulation_directory),
        "result_directory": str(result_directory),
        "runtime_seconds": np.nan,
    }

    caught_warnings: list[Any] = []
    work_lcs = None
    generative_model = None
    runresults = None
    worker_outputs = None
    stage = "validation"

    try:
        result.update(_validate_dual_lcs(lcs))
        if result["n_valid_A"] < 2 or result["n_valid_B"] < 2:
            result.update(status="insufficient_finite_points", failure_stage=stage)
            return result
        if int(n) < 1 or int(npkl) < 1:
            raise ValueError("n and npkl must both be at least one.")
        if float(truetsr) < 0.0 or float(tsrand) < 0.0:
            raise ValueError("truetsr and tsrand cannot be negative.")

        stage = "output_preparation"
        destpath.mkdir(parents=True, exist_ok=True)
        existing_paths = [
            path for path in (simulation_directory, result_directory) if path.exists()
        ]
        if existing_paths and not overwrite:
            result.update(
                status="output_already_exists",
                failure_stage=stage,
                exception_message=(
                    "Existing PyCS3 files were found. Use a unique directory "
                    "or set overwrite=True; no files were modified."
                ),
            )
            return result
        if overwrite:
            for path in existing_paths:
                shutil.rmtree(path)

        stage = "generative_fit"
        work_lcs = deepcopy(list(lcs))
        _apply_initial_shifts(
            work_lcs,
            timeshifts=initial_timeshifts,
            magshifts=initial_magshifts,
        )
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            generative_model = generative_optimizer(
                work_lcs,
                **generative_optimizer_kwargs,
            )
        caught_warnings.extend(fit_warnings)

        fitted_shift_a = float(work_lcs[0].timeshift)
        fitted_shift_b = float(work_lcs[1].timeshift)
        fitted_delay = fitted_shift_b - fitted_shift_a
        result.update(
            fitted_timeshift_A=fitted_shift_a,
            fitted_timeshift_B=fitted_shift_b,
            fitted_delay=fitted_delay,
        )
        if not np.all(np.isfinite([fitted_shift_a, fitted_shift_b, fitted_delay])):
            result.update(status="nonfinite_generative_fit", failure_stage=stage)
            return result
        result["mockable"] = True

        stage = "saving_residuals"
        pycs3.sim.draw.saveresiduals(work_lcs, generative_model)

        stage = "mock_generation"
        # Seed the parent process before drawing so the mock ensemble itself,
        # not only the worker initializations, is reproducible.
        np.random.seed(int(base_seed))
        random.seed(int(base_seed))
        with warnings.catch_warnings(record=True) as draw_warnings:
            warnings.simplefilter("always")
            pycs3.sim.draw.multidraw(
                work_lcs,
                spline=generative_model,
                onlycopy=False,
                n=int(n),
                npkl=int(npkl),
                simset=str(simset),
                shotnoise=shotnoise,
                truetsr=float(truetsr),
                tweakml=tweakml,
                tweakspl=tweakspl,
                destpath=str(destpath),
                verbose=bool(verbose),
            )
        caught_warnings.extend(draw_warnings)
        if not simulation_directory.exists():
            raise FileNotFoundError(
                f"Expected simulation directory was not created: {simulation_directory}"
            )

        stage = "mock_optimization"
        workers_used = min(max(1, int(nworkers)), int(npkl))
        result["nworkers_used"] = workers_used
        with warnings.catch_warnings(record=True) as run_warnings:
            warnings.simplefilter("always")
            worker_outputs = run_multirun_parallel(
                simset=str(simset),
                lcs=work_lcs,
                optimizer=measurement_optimizer,
                optimizer_kwargs=measurement_optimizer_kwargs,
                optset=str(optset),
                tsrand=float(tsrand),
                destpath=destpath,
                nworkers=workers_used,
                max_work_units=int(npkl),
                keepopt=False,
                base_seed=int(base_seed),
            )
        caught_warnings.extend(run_warnings)
        if not result_directory.exists():
            raise FileNotFoundError(
                f"Expected result directory was not created: {result_directory}"
            )

        stage = "result_collection"
        runresults = pycs3.sim.run.collect(
            directory=str(result_directory),
            name=str(estimator_name),
        )
        measured_shifts = np.asarray(runresults.tsarray, dtype=float)
        true_shifts = np.asarray(runresults.truetsarray, dtype=float)
        if measured_shifts.ndim != 2 or measured_shifts.shape[1] != 2:
            raise ValueError(
                f"Expected measured shift array (N, 2); got {measured_shifts.shape}."
            )
        if true_shifts.ndim != 2 or true_shifts.shape[1] != 2:
            raise ValueError(
                f"Expected true shift array (N, 2); got {true_shifts.shape}."
            )

        n_common = min(measured_shifts.shape[0], true_shifts.shape[0])
        measured_delays = (
            measured_shifts[:n_common, 1] - measured_shifts[:n_common, 0]
        )
        true_delays = true_shifts[:n_common, 1] - true_shifts[:n_common, 0]
        delay_errors = measured_delays - true_delays
        finite = (
            np.isfinite(measured_delays)
            & np.isfinite(true_delays)
            & np.isfinite(delay_errors)
        )
        finite_measured = measured_delays[finite]
        finite_true = true_delays[finite]
        finite_errors = delay_errors[finite]
        n_finite = int(np.sum(finite))
        n_failed_or_missing = max(0, requested_mocks - n_finite)
        result.update(
            n_returned=int(n_common),
            n_finite=n_finite,
            n_failed_or_missing=n_failed_or_missing,
            failure_fraction=n_failed_or_missing / requested_mocks,
        )
        if n_finite == 0:
            result.update(status="no_finite_mock_results", failure_stage=stage)
            return result

        error_p16, error_p50, error_p84 = np.percentile(
            finite_errors, [16.0, 50.0, 84.0]
        )
        measured_p16, measured_p50, measured_p84 = np.percentile(
            finite_measured, [16.0, 50.0, 84.0]
        )
        true_p16, true_p50, true_p84 = np.percentile(
            finite_true, [16.0, 50.0, 84.0]
        )
        median_bias = float(error_p50)
        random_error = float(0.5 * (error_p84 - error_p16))
        total_error = float(np.hypot(random_error, median_bias))
        
        if catastrophic_threshold_days or catastrophic_threshold_percentage is None:
            catastrophic_fraction = np.nan
        else:
            catastrophic_threshold = np.maximum(
                float(catastrophic_threshold_days),
                float(catastrophic_threshold_percentage) * np.abs(true_delays),
            )

            catastrophic_fraction = np.mean(
                np.abs(delay_errors) > catastrophic_threshold
            )

        result.update(
            status="success",
            calibrated=True,
            median_bias=median_bias,
            mean_bias=float(np.mean(finite_errors)),
            random_error=random_error,
            total_error=total_error,
            rmse=float(np.sqrt(np.mean(finite_errors**2))),
            error_p16=float(error_p16),
            error_p50=float(error_p50),
            error_p84=float(error_p84),
            measured_delay_p16=float(measured_p16),
            measured_delay_p50=float(measured_p50),
            measured_delay_p84=float(measured_p84),
            true_delay_p16=float(true_p16),
            true_delay_p50=float(true_p50),
            true_delay_p84=float(true_p84),
            catastrophic_fraction=catastrophic_fraction,
        )

        if keep_runresults:
            result.update(
                runresults=runresults,
                measured_delays=measured_delays,
                true_delays=true_delays,
                delay_errors=delay_errors,
                finite_mask=finite,
                worker_outputs=worker_outputs,
            )
        if keep_fitted_objects:
            result.update(
                fitted_lcs=work_lcs,
                generative_model=generative_model,
            )

    except Exception as exc:
        result.update(
            status=f"{stage}_failed",
            failure_stage=stage,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback=traceback.format_exc(limit=12),
        )
    finally:
        result["warnings"] = [
            f"{item.category.__name__}: {item.message}" for item in caught_warnings
        ]
        result["warning_count"] = len(result["warnings"])
        result["runtime_seconds"] = perf_counter() - start_time

    return result


def passes_mock_diagnostics(
    mock_result: Mapping[str, Any],
    *,
    minimum_finite: int = 40,
    maximum_failure_fraction: float = 0.20,
    maximum_catastrophic_fraction: float = 0.20,
) -> bool:
    """Truth-independent gate from diagnostic to production mocks."""

    if mock_result.get("status") != "success":
        return False
    if int(mock_result.get("n_finite", 0)) < int(minimum_finite):
        return False
    if float(mock_result.get("failure_fraction", np.inf)) > float(
        maximum_failure_fraction
    ):
        return False
    catastrophic = mock_result.get("catastrophic_fraction", np.nan)
    if np.isfinite(catastrophic) and float(catastrophic) > float(
        maximum_catastrophic_fraction
    ):
        return False
    return True


# ---------------------------------------------------------------------------
# Population-analysis helpers from the previous module revision
# ---------------------------------------------------------------------------


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
    """Read time, magnitude, and uncertainty arrays from a PyCS3 curve."""
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
    """Summarize one PyCS3 light curve without applying a scientific cut."""
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
    if not table.empty:
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

    table = pd.DataFrame(rows)
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
    results_table = pd.concat(result_tables, ignore_index=True)

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
        if len(data) < 3 or data[outcome].nunique() < 2:
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
        if len(data) < 3:
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
    true_abs = pd.to_numeric(table.get("true_delay"), errors="coerce").abs()

    table["relative_uncertainty_measured"] = sigma / measured_abs.replace(0.0, np.nan)
    table["relative_uncertainty_true"] = sigma / true_abs.replace(0.0, np.nan)
    table["delay_significance"] = measured_abs / sigma.replace(0.0, np.nan)

    table["truth_residual_raw"] = (
        table["observed_delay_raw"] - pd.to_numeric(table.get("true_delay"), errors="coerce")
    )
    table["truth_pull_raw"] = table["truth_residual_raw"] / sigma.replace(0.0, np.nan)

    for label, n_sigma in (("68", 1.0), ("95", 1.96)):
        table[f"ci{label}_low"] = table["observed_delay_raw"] - n_sigma * sigma
        table[f"ci{label}_high"] = table["observed_delay_raw"] + n_sigma * sigma
        true_delay = pd.to_numeric(table.get("true_delay"), errors="coerce")
        table[f"truth_covered_{label}"] = (
            true_delay.ge(table[f"ci{label}_low"])
            & true_delay.le(table[f"ci{label}_high"])
        )

    return table


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

    n_finite = pd.to_numeric(table.get("production_n_finite"), errors="coerce")
    failure_fraction = pd.to_numeric(
        table.get("observed_mock_failure_fraction"), errors="coerce"
    )
    catastrophic_fraction = pd.to_numeric(
        table.get("observed_mock_catastrophic_fraction"), errors="coerce"
    )
    sigma = pd.to_numeric(table.get("observed_mock_total_error"), errors="coerce")
    delay = pd.to_numeric(table.get("observed_delay_raw"), errors="coerce")
    relative_uncertainty = pd.to_numeric(
        table.get("relative_uncertainty_measured"), errors="coerce"
    )
    significance = pd.to_numeric(table.get("delay_significance"), errors="coerce")

    production_status = table.get(
        "production_status", pd.Series(index=table.index, dtype=object)
    )
    production_calibrated = table.get(
        "production_calibrated", pd.Series(False, index=table.index)
    ).fillna(False).astype(bool)

    table["mock_calibrated"] = (
        table["mock_status"].eq("success")
        & production_status.eq("success")
        & production_calibrated
        & n_finite.ge(minimum_finite_mocks)
        & failure_fraction.le(maximum_failure_fraction)
        & catastrophic_fraction.le(maximum_catastrophic_fraction)
    )

    table["measured_successfully"] = (
        table.get("assessable", False).fillna(False).astype(bool)
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
        residual = pd.to_numeric(table.get("truth_residual_raw"), errors="coerce")
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
        residual = pd.to_numeric(table.get("truth_residual_raw"), errors="coerce")
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

    selected.to_pickle(measurement_pickle)
    selected.drop(columns=["coords"], errors="ignore").to_csv(
        measurement_csv, index=False
    )

    with systems_pickle.open("wb") as handle:
        pickle.dump(selected_systems, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
            group.to_pickle(path)
            paths[f"band_{band_name}"] = path

    if save_per_estimator:
        for estimator, group in selected.groupby("estimator", dropna=False):
            estimator_name = str(estimator)
            path = (
                output_directory
                / f"{file_stem}_measurements_estimator_{estimator_name}.pkl"
            )
            group.to_pickle(path)
            paths[f"estimator_{estimator_name}"] = path

    return paths
