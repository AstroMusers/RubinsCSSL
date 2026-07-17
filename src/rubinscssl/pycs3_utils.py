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