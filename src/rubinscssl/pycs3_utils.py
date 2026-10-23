from __future__ import annotations

import os
from platform import system
from typing import Any
import pickle

import numpy as np
import pycs3.gen.lc_func
import pycs3.gen.mrg
import pycs3.gen.lc_func
import pycs3.spl.topopt
import pycs3.gen.mrg
import pycs3.gen.splml
import pycs3.regdiff.multiopt
import pycs3.regdiff.rslc
import pycs3.sim.draw 
import pycs3.sim.run
import pycs3.sim.plot
import pycs3.sim.twk
import pycs3.tdcomb.plot
import pycs3.tdcomb.comb

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
import shutil
import traceback
import warnings
import pycs3
import logging

from collections.abc import Mapping, Sequence
import pandas as pd

from .curve_features import (
    _curve_arrays,
    _finite_float,
    _nullable_bool,
    _summarize_curve,
    build_light_curve_feature_table_from_lcs,
    flatten_baseline_results,
    add_population_derived_columns,
    build_population_analysis_tables,
    summarize_band_performance,
    binary_feature_correlations,
    continuous_error_correlations,
    classify_mock_measurement_success,
    add_estimator_agreement,
    _as_float,
    _flatten_mock_row,
    build_mock_measurement_table,
    save_successful_time_delay_systems,
)

def make_pycs3_curve_sets(
    times: np.ndarray,
    model_magnitudes: np.ndarray,
    observed_magnitudes: np.ndarray,
    magnitude_errors: np.ndarray,
    telescope_name: str = "LSST",
) -> dict[str, Any]:
    """
    Construct matched PyCS3 curves for two lensed images.

    Returns three curve sets:

    1. ideal_model_lcs:
       Noise-free model magnitudes with realistic measurement errors.

    2. observed_lcs:
       The simulated observed magnitudes.

    The same epochs are used for all three versions of each image so their
    performances can be compared directly.
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

    ideal_model_lcs = []
    observed_lcs = []

    realized_magnitudes = np.full_like(model_magnitudes, np.nan)

    diagnostics = {}

    for image_index, image_name in enumerate(("A", "B")):
        model_mag = model_magnitudes[:, image_index]
        observed_mag = observed_magnitudes[:, image_index]
        mag_error = magnitude_errors[:, image_index]

        # Requiring observed_mag to be finite makes the model and observed
        # analyses use exactly the same detected epochs.
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

        original_indices = np.flatnonzero(valid)[order]

        ideal_lc = pycs3.gen.lc_func.factory(
            valid_times,
            valid_model_mag,
            valid_mag_error,
            telescopename=telescope_name,
            object=f"{image_name}_model_ideal",
            verbose=False,
        )


        observed_lc = pycs3.gen.lc_func.factory(
            valid_times,
            valid_observed_mag,
            valid_mag_error,
            telescopename=telescope_name,
            object=image_name,
            verbose=False,
        )

        ideal_model_lcs.append(ideal_lc)
        observed_lcs.append(observed_lc)

        diagnostics[f"n_valid_{image_name}"] = len(valid_times)
        diagnostics[f"n_removed_{image_name}"] = len(times) - len(valid_times)

        if len(valid_times) >= 2:
            diagnostics[f"baseline_{image_name}"] = (
                valid_times[-1] - valid_times[0]
            )
            diagnostics[f"median_cadence_{image_name}"] = np.median(
                np.diff(valid_times)
            )
        else:
            diagnostics[f"baseline_{image_name}"] = np.nan
            diagnostics[f"median_cadence_{image_name}"] = np.nan

        diagnostics[f"median_error_{image_name}"] = (
            np.median(valid_mag_error)
            if len(valid_mag_error) > 0
            else np.nan
        )

    for curve_set in (
        ideal_model_lcs,
        observed_lcs,
    ):
        pycs3.gen.mrg.colourise(curve_set)

    return {
        "ideal_model_lcs": ideal_model_lcs,
        "observed_lcs": observed_lcs,
        "diagnostics": diagnostics,
    }


def safe_run_pycs3_dual(
    lcs,
    optimizer,
    *,
    estimator_name,
    optimizer_kwargs=None,
    initial_timeshifts=None,
    initial_magshifts=None,
    reset_shifts=True,
    keep_objects=False,
):
    """
    Safely run one PyCS3 time-delay estimator on a double-lens system.

    Parameters
    ----------
    lcs : sequence
        Two PyCS3 LightCurve objects, ordered as [A, B].

    optimizer : callable
        Optimization function with signature:

            optimizer(lcs, **optimizer_kwargs)

        Examples are `spl` and `regdiff`.

    estimator_name : str
        Name saved in the returned result, such as "spline" or "regdiff".

    optimizer_kwargs : dict, optional
        Keyword arguments passed to the optimizer.

    initial_timeshifts : sequence of float, optional
        Initial PyCS3 time shifts for [A, B]. The default is [0, 0].

        Do not use the known true delay here. Use a fixed initialization
        policy for all systems.

    initial_magshifts : sequence of float, optional
        Initial magnitude shifts for [A, B]. The default is [0, 0].

    reset_shifts : bool
        Reset existing time and magnitude shifts before optimization.

    keep_objects : bool
        If True, include the optimized LightCurve objects and optimizer
        output in the result. Keep False for large catalog runs to avoid
        excessive memory usage.

    Returns
    -------
    dict
        A structured result that can be added to a pandas table.

        A status of "success" only means that the optimizer returned a
        finite delay. It does not mean that the delay is scientifically
        accurate.
    """

    start_time = perf_counter()
    optimizer_kwargs = dict(optimizer_kwargs or {})

    result = {
        "estimator": str(estimator_name),
        "status": "not_started",
        "converged": False,
        "delay": np.nan,
        "timeshift_A": np.nan,
        "timeshift_B": np.nan,
        "fit_metric": np.nan,
        "n_A": 0,
        "n_B": 0,
        "n_valid_A": 0,
        "n_valid_B": 0,
        "warning_count": 0,
        "warnings": [],
        "failure_stage": None,
        "exception_type": None,
        "exception_message": None,
        "traceback": None,
        "runtime_seconds": np.nan,
    }

    caught_warnings = []
    work_lcs = None
    optimizer_output = None
    stage = "validation"

    try:
        # --------------------------------------------------------------
        # 1. Minimal structural validation
        # --------------------------------------------------------------
        if not isinstance(lcs, (list, tuple)):
            raise TypeError("lcs must be a list or tuple of LightCurve objects.")

        if len(lcs) != 2:
            raise ValueError(
                f"This function expects a double with two curves; "
                f"received {len(lcs)}."
            )

        labels = ("A", "B")

        for index, label in enumerate(labels):
            lc = lcs[index]

            times = np.asarray(lc.jds, dtype=float)
            magnitudes = np.asarray(lc.mags, dtype=float)
            errors = np.asarray(lc.magerrs, dtype=float)

            if not (
                len(times) == len(magnitudes) == len(errors)
            ):
                raise ValueError(
                    f"Curve {label} has inconsistent array lengths: "
                    f"times={len(times)}, magnitudes={len(magnitudes)}, "
                    f"errors={len(errors)}."
                )

            valid = (
                np.isfinite(times)
                & np.isfinite(magnitudes)
                & np.isfinite(errors)
                & (errors > 0.0)
            )

            result[f"n_{label}"] = int(len(times))
            result[f"n_valid_{label}"] = int(np.sum(valid))
            result[f"object_{label}"] = str(
                getattr(lc, "object", label)
            )

        # This is only a structural limit, not a scientific sampling cut.
        if result["n_valid_A"] < 2 or result["n_valid_B"] < 2:
            result["status"] = "insufficient_finite_points"
            result["failure_stage"] = "validation"
            return result

        # --------------------------------------------------------------
        # 2. Make an independent copy
        # --------------------------------------------------------------
        stage = "copying_light_curves"
        work_lcs = deepcopy(list(lcs))

        # --------------------------------------------------------------
        # 3. Initialize shifts consistently
        # --------------------------------------------------------------
        stage = "initialization"

        if reset_shifts:
            for lc in work_lcs:
                lc.resetshifts()

        if initial_timeshifts is None:
            initial_timeshifts = [0.0, 0.0]

        if initial_magshifts is None:
            initial_magshifts = [0.0, 0.0]

        if len(initial_timeshifts) != 2:
            raise ValueError(
                "initial_timeshifts must contain two values for [A, B]."
            )

        if len(initial_magshifts) != 2:
            raise ValueError(
                "initial_magshifts must contain two values for [A, B]."
            )

        pycs3.gen.lc_func.applyshifts(
            work_lcs,
            timeshifts=[
                float(initial_timeshifts[0]),
                float(initial_timeshifts[1]),
            ],
            magshifts=[
                float(initial_magshifts[0]),
                float(initial_magshifts[1]),
            ],
        )

        # --------------------------------------------------------------
        # 4. Run the requested estimator
        # --------------------------------------------------------------
        stage = "optimization"

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            optimizer_output = optimizer(
                work_lcs,
                **optimizer_kwargs,
            )

        # --------------------------------------------------------------
        # 5. Extract the optimized delay
        # --------------------------------------------------------------
        stage = "delay_extraction"

        timeshift_A = float(work_lcs[0].timeshift)
        timeshift_B = float(work_lcs[1].timeshift)

        # The sign follows the ordering [A, B].
        delay_B_minus_A = timeshift_B - timeshift_A

        result["timeshift_A"] = timeshift_A
        result["timeshift_B"] = timeshift_B
        result["delay"] = float(delay_B_minus_A)

        if not np.all(
            np.isfinite(
                [
                    result["timeshift_A"],
                    result["timeshift_B"],
                    result["delay"],
                ]
            )
        ):
            result["status"] = "nonfinite_solution"
            result["failure_stage"] = "delay_extraction"
            return result

        # The spline returned by opt_fine commonly stores its final
        # fitting statistic in lastr2nostab.
        if hasattr(optimizer_output, "lastr2nostab"):
            metric = getattr(optimizer_output, "lastr2nostab")

            if np.isscalar(metric) and np.isfinite(metric):
                result["fit_metric"] = float(metric)

        # Some optimizers return a tuple whose second item may be a
        # scalar objective value.
        elif (
            isinstance(optimizer_output, tuple)
            and len(optimizer_output) > 1
            and np.isscalar(optimizer_output[1])
        ):
            try:
                metric = float(optimizer_output[1])

                if np.isfinite(metric):
                    result["fit_metric"] = metric
            except (TypeError, ValueError):
                pass

        result["status"] = "success"
        result["converged"] = True

        if keep_objects:
            result["optimized_lcs"] = work_lcs
            result["optimizer_output"] = optimizer_output

    except Exception as exc:
        result["status"] = f"{stage}_failed"
        result["failure_stage"] = stage
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        result["traceback"] = traceback.format_exc(limit=8)

    finally:
        result["warnings"] = [
            f"{item.category.__name__}: {item.message}"
            for item in caught_warnings
        ]
        result["warning_count"] = len(result["warnings"])
        result["runtime_seconds"] = perf_counter() - start_time

    return result

def assess_delay_recovery(
    run_result,
    true_delay,
    *,
    absolute_tolerance=1.0,
    relative_tolerance=0.05,
):
    """
    Compare a returned PyCS3 delay with the known simulation delay.
    """

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

    estimated_delay = float(run_result["delay"])
    true_delay = float(true_delay)

    if not np.isfinite(estimated_delay) or not np.isfinite(true_delay):
        return assessment

    error = estimated_delay - true_delay
    absolute_error = abs(error)

    tolerance = max(
        float(absolute_tolerance),
        float(relative_tolerance) * abs(true_delay),
    )

    if true_delay != 0.0:
        relative_error = absolute_error / abs(true_delay)
    else:
        relative_error = np.nan

    assessment.update(
        assessable=True,
        delay_error=error,
        absolute_delay_error=absolute_error,
        relative_delay_error=relative_error,
        tolerance=tolerance,
        accurate=absolute_error <= tolerance,
    )

    return assessment

def run_spline_multistart(
    lcs,
    spl,
    knotstep=20.0,
    maximum_shift=60.0,
    number_of_starts=17,
):
    """
    Run the spline estimator from a grid of initial delays and return
    the candidate with the smallest fit metric.
    """

    starting_delays = np.linspace(
        -maximum_shift,
        maximum_shift,
        number_of_starts,
    )

    candidates = []

    for starting_delay in starting_delays:
        result = safe_run_pycs3_dual(
            lcs,
            optimizer=spl,
            estimator_name="spline",
            optimizer_kwargs={"kn": knotstep},
            initial_timeshifts=[
                0.0,
                float(starting_delay),
            ],
        )

        result["initial_delay"] = float(starting_delay)

        if not result.get("converged", False):
            continue

        delay = result.get("delay", np.nan)
        fit_metric = result.get("fit_metric", np.nan)

        if np.isfinite(delay) and np.isfinite(fit_metric):
            candidates.append(result)

    if not candidates:
        return {
            "estimator": "spline_multistart",
            "status": "no_valid_solution",
            "converged": False,
            "delay": np.nan,
        }

    best_result = min(
        candidates,
        key=lambda result: result["fit_metric"],
    )

    best_result["estimator"] = "spline_multistart"
    best_result["n_valid_starts"] = len(candidates)
    best_result["all_candidate_delays"] = [
        candidate["delay"]
        for candidate in candidates
    ]
    best_result["all_candidate_metrics"] = [
        candidate["fit_metric"]
        for candidate in candidates
    ]

    return best_result


def run_pycs3_mocks(
    lcs,
    *,
    generative_optimizer: Callable,
    generative_optimizer_kwargs: dict | None,
    measurement_optimizer: Callable,
    measurement_optimizer_kwargs: dict | None,
    estimator_name: str,
    destpath: str | Path,
    simset: str,
    optset: str,
    n: int = 10,
    npkl: int = 5,
    truetsr: float = 10.0,
    tsrand: float = 10.0,
    shotnoise: str | None = "magerrs",
    tweakml=None,
    tweakspl=None,
    initial_timeshifts=None,
    initial_magshifts=None,
    plot_histograms: bool = False,
    catastrophic_threshold_days: float | None = 5.0,
    catastrophic_threshold_percentage: float | None = 0.20,
    overwrite: bool = False,
    keep_runresults: bool = False,
    keep_fitted_objects: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Safely generate and analyze PyCS3 mock light curves for one
    two-image system, one band, and one measurement estimator.

    The function:

    1. Fits a source spline to a fresh copy of the input curves.
    2. Generates mock curves with known true time shifts.
    3. Runs the selected time-delay estimator on all mocks.
    4. Compares recovered and true mock delays.
    5. Returns a structured calibration summary.

    Parameters
    ----------
    lcs
        Two PyCS3 LightCurve objects ordered as [A, B].

    generative_optimizer
        Function used to fit the source spline from which mocks are drawn.
        For your notebook, this will normally be `spl`.

    generative_optimizer_kwargs
        Arguments passed to the generative spline optimizer, for example
        {"kn": 20.0}.

    measurement_optimizer
        Estimator run on every mock. This can be `spl` or `regdiff`.

    measurement_optimizer_kwargs
        Arguments passed to the measurement optimizer.

    estimator_name
        Descriptive estimator name saved in the output.

    destpath
        Unique directory for this system, band, curve type, and run.

    simset
        Mock-set name, such as "observed_mock_diagnostic".

    optset
        Estimator output name, such as "spl" or "reg".

    n, npkl
        PyCS3 writes `npkl` pickle files containing `n` mock sets each.
        Therefore, the requested number of mocks is `n * npkl`.

    truetsr
        Radius in days over which true mock shifts are randomized around
        the fitted shifts.

    tsrand
        Radius in days for randomizing the estimator's initial shifts.

    shotnoise
        Noise option passed to multidraw. For a first controlled test,
        use "magerrs". Use "mcres" only when residual-based noise is
        scientifically justified.

    tweakml, tweakspl
        Optional PyCS3 mock-perturbation functions.

    overwrite
        If True, remove existing simulation and result directories first.
        If False, refuse to mix the new run with old files.

    Returns
    -------
    dict
        Structured status, diagnostics, and mock-calibration statistics.

    Notes
    -----
    `status == "success"` means the wrapper completed and at least one
    finite mock result was recovered. It does not by itself imply that
    the calibration is scientifically acceptable.
    """

    start_time = perf_counter()

    generative_optimizer_kwargs = dict(
        generative_optimizer_kwargs or {}
    )
    measurement_optimizer_kwargs = dict(
        measurement_optimizer_kwargs or {}
    )

    destpath = Path(destpath)
    simulation_directory = destpath / f"sims_{simset}"
    result_directory = destpath / f"sims_{simset}_opt_{optset}"

    requested_mocks = int(n) * int(npkl)

    result = {
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
        "catastrophic_threshold_days": (
            catastrophic_threshold_days
        ),
        "catastrophic_threshold_percentage": (
            catastrophic_threshold_percentage
        ),
        "catastrophic_fraction": np.nan,
        "n_valid_A": 0,
        "n_valid_B": 0,
        "fitted_timeshift_A": np.nan,
        "fitted_timeshift_B": np.nan,
        "fitted_delay": np.nan,
        "warning_count": 0,
        "warnings": [],
        "exception_type": None,
        "exception_message": None,
        "traceback": None,
        "simulation_directory": str(simulation_directory),
        "result_directory": str(result_directory),
        "runtime_seconds": np.nan,
    }

    caught_warnings = []
    work_lcs = None
    generative_spline = None
    runresults = None
    stage = "validation"

    try:
        # ----------------------------------------------------------
        # 1. Validate the request
        # ----------------------------------------------------------
        if not isinstance(lcs, (list, tuple)):
            raise TypeError(
                "lcs must be a list or tuple of LightCurve objects."
            )

        if len(lcs) != 2:
            raise ValueError(
                "This wrapper currently expects two curves ordered "
                f"as [A, B]; received {len(lcs)} curves."
            )

        if n < 1 or npkl < 1:
            raise ValueError("n and npkl must both be at least 1.")

        if truetsr < 0.0:
            raise ValueError("truetsr cannot be negative.")

        if tsrand < 0.0:
            raise ValueError("tsrand cannot be negative.")

        for index, label in enumerate(("A", "B")):
            lc = lcs[index]

            times = np.asarray(lc.jds, dtype=float)
            mags = np.asarray(lc.mags, dtype=float)
            errors = np.asarray(lc.magerrs, dtype=float)

            if not (
                len(times) == len(mags) == len(errors)
            ):
                raise ValueError(
                    f"Curve {label} has inconsistent array lengths."
                )

            valid = (
                np.isfinite(times)
                & np.isfinite(mags)
                & np.isfinite(errors)
                & (errors > 0.0)
            )

            result[f"n_valid_{label}"] = int(np.sum(valid))

        # This is a structural requirement, not a scientific quality cut.
        if result["n_valid_A"] < 2 or result["n_valid_B"] < 2:
            result["status"] = "insufficient_finite_points"
            result["failure_stage"] = "validation"
            return result

        # ----------------------------------------------------------
        # 2. Protect against stale PyCS3 output
        # ----------------------------------------------------------
        stage = "output_preparation"
        destpath.mkdir(parents=True, exist_ok=True)

        existing_paths = [
            path
            for path in (
                simulation_directory,
                result_directory,
            )
            if path.exists()
        ]

        if existing_paths and not overwrite:
            result["status"] = "output_already_exists"
            result["failure_stage"] = "output_preparation"
            result["exception_message"] = (
                "Existing PyCS3 mock files were found. Use a unique "
                "run directory or set overwrite=True. Existing files "
                "were not modified."
            )
            return result

        if overwrite:
            for path in existing_paths:
                shutil.rmtree(path)

        # Remove abandoned parallel-processing markers if the main
        # result directory has just been recreated or cleaned.
        if result_directory.exists():
            for marker in result_directory.glob("*.workingon"):
                marker.unlink()
    


        # ----------------------------------------------------------
        # 3. Fit the generative source spline
        # ----------------------------------------------------------
        stage = "generative_fit"
        work_lcs = deepcopy(list(lcs))

        for lc in work_lcs:
            lc.resetshifts()

        if initial_timeshifts is None:
            initial_timeshifts = [0.0, 0.0]

        if initial_magshifts is None:
            initial_magshifts = [0.0, 0.0]

        if len(initial_timeshifts) != 2:
            raise ValueError(
                "initial_timeshifts must contain two values."
            )

        if len(initial_magshifts) != 2:
            raise ValueError(
                "initial_magshifts must contain two values."
            )

        pycs3.gen.lc_func.applyshifts(
            work_lcs,
            timeshifts=[
                float(initial_timeshifts[0]),
                float(initial_timeshifts[1]),
            ],
            magshifts=[
                float(initial_magshifts[0]),
                float(initial_magshifts[1]),
            ],
        )

        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")

            generative_spline = generative_optimizer(
                work_lcs,
                **generative_optimizer_kwargs,
            )

        caught_warnings.extend(fit_warnings)

        fitted_shift_A = float(work_lcs[0].timeshift)
        fitted_shift_B = float(work_lcs[1].timeshift)
        fitted_delay = fitted_shift_B - fitted_shift_A

        result.update(
            fitted_timeshift_A=fitted_shift_A,
            fitted_timeshift_B=fitted_shift_B,
            fitted_delay=fitted_delay,
        )

        if not np.all(
            np.isfinite(
                [
                    fitted_shift_A,
                    fitted_shift_B,
                    fitted_delay,
                ]
            )
        ):
            result["status"] = "nonfinite_generative_fit"
            result["failure_stage"] = "generative_fit"
            return result

        result["mockable"] = True

        # PyCS3 needs saved residuals for residual-based shot-noise
        # options such as "mcres". Saving them is harmless for
        # shotnoise="magerrs" and keeps the workflow consistent.
        stage = "saving_residuals"

        pycs3.sim.draw.saveresiduals(
            work_lcs,
            generative_spline,
        )
        # # ----------------------------------------------------------
        # # 3. Drawing copies to inspect intrinsic variability and noise is not yet implemented.
        # # ----------------------------------------------------------

        # stage = "drawing_copies"
        # copies_spline = generative_spline.copy()
        # with warnings.catch_warnings(record=True) as draw_warnings:
        #     warnings.simplefilter("always")

        #     pycs3.sim.draw.multidraw(
        #         copies_spline,
        #         onlycopy=True,
        #         n=int(n),
        #         npkl=int(npkl),
        #         simset=str(simset)+"_copies",
        #         destpath=str(destpath),
        #         verbose=bool(verbose),
        #     )

        # caught_warnings.extend(draw_warnings)

        # if not simulation_directory.exists():
        #     raise FileNotFoundError(
        #         "PyCS3 did not create the expected simulation "
        #         f"directory: {simulation_directory}"
        #     )   
        
        # # ----------------------------------------------------------
        # # 5. Measure the delays in every copy
        # # ----------------------------------------------------------
        # stage = "copy_optimization"

        # with warnings.catch_warnings(record=True) as run_warnings:
        #     warnings.simplefilter("always")

        #     pycs3.sim.run.multirun(
        #         str(simset)+"_copies",
        #         work_lcs,
        #         measurement_optimizer,
        #         kwargs_optim=measurement_optimizer_kwargs,
        #         optset=str(optset),
        #         tsrand=float(tsrand),
        #         keepopt=False,
        #         destpath=str(destpath),
        #     )

        # caught_warnings.extend(run_warnings)

        # if not result_directory.exists():
        #     raise FileNotFoundError(
        #         "PyCS3 did not create the expected result directory: "
        #         f"{result_directory}"
        #     )
        
        # # ----------------------------------------------------------
        # # 5. Measure the delays in every copy
        # # ----------------------------------------------------------
        # stage = "copy_optimization"

        # with warnings.catch_warnings(record=True) as run_warnings:
        #     warnings.simplefilter("always")

        #     pycs3.sim.run.multirun(
        #         str(simset)+"_copies",
        #         work_lcs,
        #         measurement_optimizer,
        #         kwargs_optim=measurement_optimizer_kwargs,
        #         optset=str(optset),
        #         tsrand=float(tsrand),
        #         keepopt=False,
        #         destpath=str(destpath),
        #     )

        # caught_warnings.extend(run_warnings)

        # if not result_directory.exists():
        #     raise FileNotFoundError(
        #         "PyCS3 did not create the expected result directory: "
        #         f"{result_directory}"
        #     )
        

        # ----------------------------------------------------------
        # 4. Draw the mock curves
        # ----------------------------------------------------------
        stage = "mock_generation"

        with warnings.catch_warnings(record=True) as draw_warnings:
            warnings.simplefilter("always")

            pycs3.sim.draw.multidraw(
                work_lcs,
                spline=generative_spline,
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
                "PyCS3 did not create the expected simulation "
                f"directory: {simulation_directory}"
            )

        # ----------------------------------------------------------
        # 5. Measure the delays in every mock
        # ----------------------------------------------------------
        stage = "mock_optimization"

        with warnings.catch_warnings(record=True) as run_warnings:
            warnings.simplefilter("always")

            pycs3.sim.run.multirun(
                str(simset),
                work_lcs,
                measurement_optimizer,
                kwargs_optim=measurement_optimizer_kwargs,
                optset=str(optset),
                tsrand=float(tsrand),
                keepopt=False,
                destpath=str(destpath),
            )

        caught_warnings.extend(run_warnings)

        if not result_directory.exists():
            raise FileNotFoundError(
                "PyCS3 did not create the expected result directory: "
                f"{result_directory}"
            )

        # ----------------------------------------------------------
        # 6. Collect and analyze recovered-versus-true shifts
        # ----------------------------------------------------------
        stage = "result_collection"

        runresults = pycs3.sim.run.collect(
            directory=str(result_directory),
            name=str(estimator_name),
        )

        if plot_histograms:
            pycs3.sim.plot.hists(runresults, 
            r=5.0, nbins=100, showqs=False, 
            dataout=True, usemedian=True, 
            outdir=result_directory,
            filename=os.path.join(result_directory, f"{estimator_name}_mock_delay_histograms.png"))

            pycs3.gen.stat.anaoptdrawn(work_lcs, 
            generative_spline, simset=str(simset), 
            optset=str(optset), 
            showplot=False,
            nplots=1,
            directory=str(result_directory),
            plotpath=os.path.join(result_directory, f"{estimator_name}_mock_delay_analysis.png"))

            pycs3.sim.plot.measvstrue(runresults, 
            errorrange=3.5, 
            r=5.0, 
            nbins = 1, 
            binclip=True, 
            binclipr=20.0,
            plotpoints=True, 
            dataout=True, 
            outdir=result_directory,
            filename=os.path.join(result_directory, f"{estimator_name}_mock_measured_vs_true.png"))

        measured_shifts = np.asarray(
            runresults.tsarray,
            dtype=float,
        )
        true_shifts = np.asarray(
            runresults.truetsarray,
            dtype=float,
        )

        if measured_shifts.ndim != 2:
            raise ValueError(
                "Collected measured shifts are not a two-dimensional "
                f"array: shape={measured_shifts.shape}."
            )

        if true_shifts.ndim != 2:
            raise ValueError(
                "Collected true shifts are not a two-dimensional "
                f"array: shape={true_shifts.shape}."
            )

        if measured_shifts.shape[1] != 2:
            raise ValueError(
                "Expected two measured shift columns for [A, B], "
                f"received shape {measured_shifts.shape}."
            )

        if true_shifts.shape[1] != 2:
            raise ValueError(
                "Expected two true shift columns for [A, B], "
                f"received shape {true_shifts.shape}."
            )

        n_common = min(
            measured_shifts.shape[0],
            true_shifts.shape[0],
        )

        measured_shifts = measured_shifts[:n_common]
        true_shifts = true_shifts[:n_common]

        measured_delays = (
            measured_shifts[:, 1]
            - measured_shifts[:, 0]
        )

        true_delays = (
            true_shifts[:, 1]
            - true_shifts[:, 0]
        )

        delay_errors = measured_delays - true_delays

        finite = (
            np.isfinite(measured_delays)
            & np.isfinite(true_delays)
            & np.isfinite(delay_errors)
        )

        finite_measured = measured_delays[finite]
        finite_true = true_delays[finite]
        finite_errors = delay_errors[finite]

        n_returned = int(n_common)
        n_finite = int(np.sum(finite))

        # Include missing result rows in the failure fraction.
        n_failed_or_missing = max(
            0,
            requested_mocks - n_finite,
        )

        result.update(
            n_returned=n_returned,
            n_finite=n_finite,
            n_failed_or_missing=n_failed_or_missing,
            failure_fraction=(
                n_failed_or_missing / requested_mocks
            ),
        )

        if n_finite == 0:
            result["status"] = "no_finite_mock_results"
            result["failure_stage"] = "result_collection"
            return result

        error_p16, error_p50, error_p84 = np.percentile(
            finite_errors,
            [16.0, 50.0, 84.0],
        )

        measured_p16, measured_p50, measured_p84 = (
            np.percentile(
                finite_measured,
                [16.0, 50.0, 84.0],
            )
        )

        true_p16, true_p50, true_p84 = np.percentile(
            finite_true,
            [16.0, 50.0, 84.0],
        )

        median_bias = float(error_p50)
        mean_bias = float(np.mean(finite_errors))

        random_error = float(
            0.5 * (error_p84 - error_p16)
        )

        rmse = float(
            np.sqrt(np.mean(finite_errors**2))
        )

        total_error = float(
            np.sqrt(
                random_error**2
                + median_bias**2
            )
        )

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
            mean_bias=mean_bias,
            random_error=random_error,
            total_error=total_error,
            rmse=rmse,
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

        # Arrays are helpful for detailed plots but should generally not
        # be inserted directly into a flat pandas results table.
        if keep_runresults:
            result["runresults"] = runresults
            result["measured_delays"] = measured_delays
            result["true_delays"] = true_delays
            result["delay_errors"] = delay_errors
            result["finite_mask"] = finite

        if keep_fitted_objects:
            result["fitted_lcs"] = work_lcs
            result["generative_spline"] = generative_spline

    except Exception as exc:
        result["status"] = f"{stage}_failed"
        result["failure_stage"] = stage
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        result["traceback"] = traceback.format_exc(limit=10)

    finally:
        result["warnings"] = [
            f"{item.category.__name__}: {item.message}"
            for item in caught_warnings
        ]
        result["warning_count"] = len(result["warnings"])
        result["runtime_seconds"] = perf_counter() - start_time

    return result

def passes_mock_diagnostics(
    mock_result,
    *,
    minimum_finite=40,
    maximum_failure_fraction=0.20,
    maximum_catastrophic_fraction=0.20,
):
    if mock_result["status"] != "success":
        return False

    if mock_result["n_finite"] < minimum_finite:
        return False

    if (
        mock_result["failure_fraction"]
        > maximum_failure_fraction
    ):
        return False

    catastrophic_fraction = mock_result[
        "catastrophic_fraction"
    ]

    if (
        np.isfinite(catastrophic_fraction)
        and catastrophic_fraction
        > maximum_catastrophic_fraction
    ):
        return False

    return True

import numpy as np
import pandas as pd


def _first_scalar(value, default=np.nan):
    """Return the first finite scalar contained in an array-like value."""
    try:
        array = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return default

    return float(array[0]) if array.size else default


def _summarize_single_curve(times, magnitudes, errors):
    """
    Return sampling and uncertainty diagnostics for one image.
    """

    times = np.asarray(times, dtype=float)
    magnitudes = np.asarray(magnitudes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    valid = (
        np.isfinite(times)
        & np.isfinite(magnitudes)
        & np.isfinite(errors)
        & (errors > 0.0)
    )

    times = np.sort(times[valid])
    magnitudes = magnitudes[valid]
    errors = errors[valid]

    n_points = len(times)

    if n_points == 0:
        return {
            "n": 0,
            "baseline": np.nan,
            "median_cadence": np.nan,
            "max_gap": np.nan,
            "median_error": np.nan,
            "median_magnitude": np.nan,
        }

    if n_points == 1:
        baseline = 0.0
        median_cadence = np.nan
        max_gap = np.nan
    else:
        gaps = np.diff(times)
        baseline = times[-1] - times[0]
        median_cadence = np.median(gaps)
        max_gap = np.max(gaps)

    return {
        "n": int(n_points),
        "baseline": float(baseline),
        "median_cadence": float(median_cadence),
        "max_gap": float(max_gap),
        "median_error": float(np.median(errors)),
        "median_magnitude": float(np.median(magnitudes)),
    }


def build_light_curve_feature_table(
    raw_systems,
    *,
    curve_type="observed",
    bands=("g", "r", "i"),
):
    """
    Build one feature row per system and band.

    Parameters
    ----------
    raw_systems
        Your `dual_lens_systems` dictionary.

    curve_type
        "observed" uses obs_mag.
        "model" uses model_mag.

    Returns
    -------
    pandas.DataFrame
    """

    if curve_type == "observed":
        magnitude_key = "obs_mag"
    elif curve_type == "model":
        magnitude_key = "model_mag"
    else:
        raise ValueError("curve_type must be 'observed' or 'model'.")

    rows = []

    for system_id, system in raw_systems.items():
        try:
            visit_bands = np.asarray(
                system["obs_bands"][0]
            ).astype(str)

            relative_times = np.asarray(
                system["obs_times"][0],
                dtype=float,
            )

            obs_start = _first_scalar(system["obs_start"], default=0.0)
            times = relative_times + obs_start

            magnitudes = np.asarray(
                system[magnitude_key][0],
                dtype=float,
            )

            errors = np.asarray(
                system["obs_mag_error"][0],
                dtype=float,
            )

        except (KeyError, IndexError, TypeError, ValueError) as exc:
            rows.append(
                {
                    "system_id": system_id,
                    "band": None,
                    "curve_type": curve_type,
                    "feature_status": "construction_failed",
                    "feature_exception": repr(exc),
                }
            )
            continue

        if magnitudes.ndim != 2 or magnitudes.shape[1] != 2:
            rows.append(
                {
                    "system_id": system_id,
                    "band": None,
                    "curve_type": curve_type,
                    "feature_status": "invalid_magnitude_shape",
                }
            )
            continue

        for band in bands:
            band_mask = visit_bands == band

            band_times = times[band_mask]
            band_magnitudes = magnitudes[band_mask]
            band_errors = errors[band_mask]

            if band_magnitudes.size == 0:
                rows.append(
                    {
                        "system_id": system_id,
                        "band": band,
                        "curve_type": curve_type,
                        "feature_status": "band_unavailable",
                        "n_band_visits": 0,
                        "n_A": 0,
                        "n_B": 0,
                        "n_min": 0,
                        "n_total": 0,
                        "n_shared": 0,
                        "shared_fraction": 0.0,
                    }
                )
                continue

            valid_A = (
                np.isfinite(band_times)
                & np.isfinite(band_magnitudes[:, 0])
                & np.isfinite(band_errors[:, 0])
                & (band_errors[:, 0] > 0.0)
            )

            valid_B = (
                np.isfinite(band_times)
                & np.isfinite(band_magnitudes[:, 1])
                & np.isfinite(band_errors[:, 1])
                & (band_errors[:, 1] > 0.0)
            )

            stats_A = _summarize_single_curve(
                band_times,
                band_magnitudes[:, 0],
                band_errors[:, 0],
            )

            stats_B = _summarize_single_curve(
                band_times,
                band_magnitudes[:, 1],
                band_errors[:, 1],
            )

            times_A = np.sort(band_times[valid_A])
            times_B = np.sort(band_times[valid_B])

            if len(times_A) and len(times_B):
                overlap_start = max(times_A[0], times_B[0])
                overlap_end = min(times_A[-1], times_B[-1])
                overlap_baseline = max(
                    0.0,
                    overlap_end - overlap_start,
                )
            else:
                overlap_baseline = 0.0

            n_shared = int(np.sum(valid_A & valid_B))
            n_union = int(np.sum(valid_A | valid_B))

            median_cadence = np.nanmean(
                [
                    stats_A["median_cadence"],
                    stats_B["median_cadence"],
                ]
            )

            max_gap = np.nanmax(
                [
                    stats_A["max_gap"],
                    stats_B["max_gap"],
                ]
            )

            median_error = np.nanmedian(
                [
                    stats_A["median_error"],
                    stats_B["median_error"],
                ]
            )

            rows.append(
                {
                    "system_id": system_id,
                    "band": band,
                    "curve_type": curve_type,
                    "feature_status": "success",
                    "n_band_visits": int(np.sum(band_mask)),
                    "n_A": stats_A["n"],
                    "n_B": stats_B["n"],
                    "n_min": min(stats_A["n"], stats_B["n"]),
                    "n_total": stats_A["n"] + stats_B["n"],
                    "n_shared": n_shared,
                    "shared_fraction": (
                        n_shared / n_union
                        if n_union > 0
                        else 0.0
                    ),
                    "baseline_A": stats_A["baseline"],
                    "baseline_B": stats_B["baseline"],
                    "baseline_min": np.nanmin(
                        [
                            stats_A["baseline"],
                            stats_B["baseline"],
                        ]
                    ),
                    "overlap_baseline": overlap_baseline,
                    "median_cadence_A": stats_A["median_cadence"],
                    "median_cadence_B": stats_B["median_cadence"],
                    "median_cadence": median_cadence,
                    "max_gap_A": stats_A["max_gap"],
                    "max_gap_B": stats_B["max_gap"],
                    "max_gap": max_gap,
                    "median_error_A": stats_A["median_error"],
                    "median_error_B": stats_B["median_error"],
                    "median_error": median_error,
                    "median_magnitude_A": stats_A["median_magnitude"],
                    "median_magnitude_B": stats_B["median_magnitude"],
                    "median_image_contrast": (
                        stats_B["median_magnitude"]
                        - stats_A["median_magnitude"]
                    ),
                    "z_lens": _first_scalar(
                        system.get("z_lens", np.nan)
                    ),
                    "z_source": _first_scalar(
                        system.get("z_source", np.nan)
                    ),
                    "einstein_radius": _first_scalar(
                        system.get("theta_E", np.nan)
                    ),
                    "batch": system.get("batch", np.nan),
                }
            )

    return pd.DataFrame(rows)
