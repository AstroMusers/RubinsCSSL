from __future__ import annotations

from dataclasses import dataclass
from logging import config
from typing import Any, Callable, Mapping

from matplotlib.table import table
import numpy as np
import pandas as pd

from astropy.table import Table
from dynesty import utils as dyfunc

from gaussn import utils


@dataclass(frozen=True)
class DegradationConfig:
    # Astropy Table column names
    time_col: str = "time"
    flux_col: str = "flux"
    fluxerr_col: str = "fluxerr"
    band_col: str = "band"
    image_col: str = "image"

    # For fitting_SLSN_example.ipynb:
    # [A, tau, t0, x0, x1, c, delay, magnification]
    delay_index: int = 2

    # Number of independent random removal orders per system
    n_paths: int = 20

    # 1 means fit after every removed row.
    # Use 2, 5, or 10 for a less expensive pilot.
    fit_every: int = 1

    n_expected_images: int = 2

    # Backward-compatible cap on removed items.
    # Prefer max_removed_groups for grouped exposure removal.
    max_removed_points: int | None = None
    max_removed_groups: int | None = None

    master_seed: int = 20260716

    # Conservative structural requirements.
    # These are experiment choices, not GausSN requirements.
    min_points_per_image: int = 4
    min_points_per_shared_band_image: int = 3

    # The same delay prior must be used at every thinning level.
    delay_prior_min: float = 0.0
    delay_prior_max: float = 60.0

    # Posterior-quality definition
    max_ci68_fraction_of_prior: float = 0.80
    prior_edge_fraction: float = 0.02

    # Do not stop after one possibly stochastic sampler failure.
    max_consecutive_fit_failures: int = 3

def prepare_for_gaussn(
    table: Table,
    config: DegradationConfig,  
    ) -> Table:
    """Return a copied and correctly sorted table."""
    output = table.copy(copy_data=True)

    output.sort(
        [
            config.band_col,
            config.image_col,
            config.time_col,
        ]
    )

    return output


def structural_status(
    table: Table,
    config: DegradationConfig,
    ) -> tuple[bool, str, dict]:
    """
    Check whether a two-image subset still has enough structure
    to attempt a meaningful delay fit.
    """
    time = np.asarray(table[config.time_col], dtype=float)
    flux = np.asarray(table[config.flux_col], dtype=float)
    fluxerr = np.asarray(table[config.fluxerr_col], dtype=float)
    band = np.asarray(table[config.band_col])
    image = np.asarray(table[config.image_col])

    metadata = {
        "n_shared_informative_bands": 0,
        "minimum_image_count": 0,
    }

    if len(table) == 0:
        return False, "empty_table", metadata

    if not (
        np.all(np.isfinite(time))
        and np.all(np.isfinite(flux))
        and np.all(np.isfinite(fluxerr))
    ):
        return False, "nonfinite_values", metadata

    if np.any(fluxerr <= 0):
        return False, "nonpositive_flux_error", metadata

    if np.ptp(flux) <= 0:
        return False, "zero_flux_range", metadata

    images = np.unique(image)

    # This experiment assumes the doubly imaged notebook model.
    if len(images) != config.n_expected_images:
        return False, f"expected_{config.n_expected_images}_images_found_{len(images)}", metadata

    image_counts = {
        image_id: int(np.sum(image == image_id))
        for image_id in images
    }

    metadata["minimum_image_count"] = min(image_counts.values())

    if metadata["minimum_image_count"] < config.min_points_per_image:
        return False, "too_few_points_in_one_image", metadata

    bands_by_image = {
        image_id: set(np.unique(band[image == image_id]))
        for image_id in images
    }

    shared_bands = set.intersection(
        *(bands_by_image[image_id] for image_id in images)
    )

    informative_bands = []

    for band_id in shared_bands:
        counts = [
            np.sum((band == band_id) & (image == image_id))
            for image_id in images
        ]

        if min(counts) >= config.min_points_per_shared_band_image:
            informative_bands.append(band_id)

    metadata["n_shared_informative_bands"] = len(
        informative_bands
    )

    if len(informative_bands) == 0:
        return False, "no_informative_shared_band", metadata

    return True, "ok", metadata

def summarize_delay_posterior(
    sampler: Any,
    truth_delay: float,
    config: DegradationConfig,
) -> dict:
    """Summarize one GausSN Dynesty time-delay posterior."""
    results = sampler.results

    samples = np.asarray(results.samples, dtype=float)
    delays = samples[:, config.delay_index]

    if hasattr(results, "importance_weights"):
        weights = np.asarray(
            results.importance_weights(),
            dtype=float,
        )
    else:
        weights = np.exp(
            np.asarray(results.logwt)
            - float(results.logz[-1])
        )

    weights /= np.sum(weights)

    q025, q16, q50, q84, q975 = dyfunc.quantile(
        delays,
        [0.025, 0.16, 0.50, 0.84, 0.975],
        weights=weights,
    )

    posterior_mean = np.sum(weights * delays)

    posterior_variance = np.sum(
        weights * (delays - posterior_mean) ** 2
    )
    posterior_std = np.sqrt(posterior_variance)

    # Importance-sampling effective sample size
    posterior_ess = 1.0 / np.sum(weights**2)

    signed_error = q50 - truth_delay
    absolute_error = abs(signed_error)

    if truth_delay != 0:
        fractional_error = signed_error / abs(truth_delay)
        absolute_fractional_error = (
            absolute_error / abs(truth_delay)
        )
    else:
        fractional_error = np.nan
        absolute_fractional_error = np.nan

    ci68_width = q84 - q16
    ci95_width = q975 - q025
    precision_half_width = 0.5 * ci68_width

    if truth_delay != 0:
        fractional_precision = (
            precision_half_width / abs(truth_delay)
        )
    else:
        fractional_precision = np.nan

    # Use the uncertainty extending toward the truth.
    if q50 >= truth_delay:
        pull_scale = q50 - q16
    else:
        pull_scale = q84 - q50

    pull = (
        signed_error / pull_scale
        if pull_scale > 0
        else np.nan
    )

    prior_width = (
        config.delay_prior_max
        - config.delay_prior_min
    )
    edge_margin = config.prior_edge_fraction * prior_width

    touches_prior_edge = bool(
        (q16 <= config.delay_prior_min + edge_margin)
        or
        (q84 >= config.delay_prior_max - edge_margin)
    )

    constrained = bool(
        np.isfinite(q50)
        and ci68_width
        < config.max_ci68_fraction_of_prior * prior_width
        and not touches_prior_edge
    )

    return {
        "truth_delay_days": float(truth_delay),
        "delay_median_days": float(q50),
        "delay_mean_days": float(posterior_mean),

        "q025_days": float(q025),
        "q16_days": float(q16),
        "q84_days": float(q84),
        "q975_days": float(q975),

        "ci68_width_days": float(ci68_width),
        "ci95_width_days": float(ci95_width),
        "posterior_std_days": float(posterior_std),
        "posterior_ess": float(posterior_ess),

        "signed_error_days": float(signed_error),
        "absolute_error_days": float(absolute_error),
        "fractional_error": float(fractional_error),
        "absolute_fractional_error": float(
            absolute_fractional_error
        ),
        "fractional_precision": float(
            fractional_precision
        ),

        "pull": float(pull),
        "covered_68": bool(q16 <= truth_delay <= q84),
        "covered_95": bool(q025 <= truth_delay <= q975),

        "touches_prior_edge": touches_prior_edge,
        "constrained": constrained,

        "accuracy_below_5_percent": bool(
            np.isfinite(absolute_fractional_error)
            and absolute_fractional_error < 0.05
        ),
        "accuracy_below_10_percent": bool(
            np.isfinite(absolute_fractional_error)
            and absolute_fractional_error < 0.10
        ),
    }

def make_seed(master_seed: int, *parts: int) -> int:
    sequence = np.random.SeedSequence(
        [master_seed, *parts]
    )
    return int(sequence.generate_state(1)[0])


def run_degradation_experiment(
    systems: Mapping[str, Table],
    truth_delays: Mapping[str, float],
    fit_function: Callable[[Table, int], Any],
    config: DegradationConfig,
) -> pd.DataFrame:
    records = []

    sorted_system_ids = sorted(systems)

    for system_number, system_id in enumerate(
        sorted_system_ids
    ):
        original = systems[system_id].copy(copy_data=True)

        # Persistent identifier unaffected by sorting or deletion.
        original["_row_id"] = np.arange(
            len(original),
            dtype=int,
        )

        original = prepare_for_gaussn(original, config)

        truth_delay = float(truth_delays[system_id])
           
        n_initial = len(original)
        max_removed = (
            config.max_removed_groups
            if config.max_removed_groups is not None
            else config.max_removed_points
        )

        if max_removed is None:
            n_max_removed = n_initial
        else:
            if max_removed < 0:
                raise ValueError(
                    "max_removed_groups must be a non-negative integer or None"
                )
            n_max_removed = min(max_removed, n_initial)

        eligible, reason, structure = structural_status(
            original,
            config,
        )

        if not eligible:
            records.append(
                {
                    "system_id": system_id,
                    "path_id": -1,
                    "n_initial": n_initial,
                    "n_remaining": n_initial,
                    "n_removed": 0,
                    "retained_fraction": 1.0,
                    "fit_success": False,
                    "constrained": False,
                    "failure_reason": reason,
                    **structure,
                }
            )
            continue

        # Fit complete table exactly once.
        baseline_seed = make_seed(
            config.master_seed,
            system_number,
            0,
            0,
        )

        try:
            baseline_sampler = fit_function(
                original,
                baseline_seed,
            )

            baseline_summary = summarize_delay_posterior(
                baseline_sampler,
                truth_delay,
                config,
            )

            baseline_success = True
            baseline_reason = "ok"

        except Exception as error:
            baseline_success = False
            baseline_reason = (
                f"{type(error).__name__}: {error}"
            )
            baseline_summary = {
                "constrained": False,
            }

        if not baseline_success:
            records.append(
                {
                    "system_id": system_id,
                    "path_id": -1,
                    "n_initial": n_initial,
                    "n_remaining": n_initial,
                    "n_removed": 0,
                    "retained_fraction": 1.0,
                    "fit_success": False,
                    "failure_reason": baseline_reason,
                    **structure,
                }
            )
            continue

        # Independent nested removal paths
        for path_id in range(config.n_paths):
            # Reuse the baseline result as step zero for every path.
            records.append(
                {
                    "system_id": system_id,
                    "path_id": path_id,
                    "n_initial": n_initial,
                    "n_remaining": n_initial,
                    "n_removed": 0,
                    "retained_fraction": 1.0,

                    "removed_row_id": np.nan,
                    "removed_time": np.nan,
                    "removed_flux": np.nan,
                    "removed_fluxerr": np.nan,
                    "removed_snr": np.nan,
                    "removed_band": None,
                    "removed_image": None,

                    "fit_success": True,
                    "failure_reason": "ok",
                    "reused_baseline": True,

                    **structure,
                    **baseline_summary,
                }
            )

            path_seed = make_seed(
                config.master_seed,
                system_number,
                path_id,
                1,
            )
            path_rng = np.random.default_rng(path_seed)

            row_ids = np.asarray(
                original["_row_id"],
                dtype=int,
            )
            # Find row ids sharing the same time stamp, and group them.
            rows_same_exposure = {}
            for row in original:
                time = row[config.time_col]
                if time not in rows_same_exposure:
                    rows_same_exposure[time] = []
                rows_same_exposure[time].append(row["_row_id"])
            exposure_groups = list(rows_same_exposure.values())



            removal_order = list(exposure_groups)
            path_rng.shuffle(removal_order)

            consecutive_failures = 0

            for n_removed in range(1, n_max_removed + 1):
                removed_groups = removal_order[:n_removed]
                removed_ids = [row_id for group in removed_groups for row_id in group]
                keep_mask = ~np.isin(
                    np.asarray(original["_row_id"]),
                    removed_ids,
                )
                if not np.any(keep_mask):
                    print("No rows remaining after removal.")
                    break
                subset = original[keep_mask]
                subset = prepare_for_gaussn(
                    subset,
                    config,
                )

                n_remaining = len(subset)

                # Metadata for the row group removed at this step
                current_removed_group = removal_order[
                    n_removed - 1
                ]

                removed_rows = original[
                    np.isin(
                        np.asarray(original["_row_id"]),
                        current_removed_group,
                    )
                ]

                removed_flux = [
                    float(removed_row[config.flux_col])
                    for removed_row in removed_rows
                ]
                removed_fluxerr = [
                    float(removed_row[config.fluxerr_col])
                    for removed_row in removed_rows
                ]
                removed_time = (
                    float(removed_rows[config.time_col][0])
                    if len(removed_rows) > 0
                    else np.nan
                )
                removed_bands = [
                    str(removed_row[config.band_col])
                    for removed_row in removed_rows
                ]
                removed_images = [
                    str(removed_row[config.image_col])
                    for removed_row in removed_rows
                ]
                removed_snr = [
                    abs(f) / fe if fe > 0 else np.nan
                    for f, fe in zip(removed_flux, removed_fluxerr)
                ]

                metadata = {
                    "system_id": system_id,
                    "path_id": path_id,
                    "n_initial": n_initial,
                    "n_remaining": n_remaining,
                    "n_removed": n_removed,
                    "retained_fraction": (
                        n_remaining / n_initial
                    ),
                    "removed_group_ids": list(removed_ids),
                    "removed_row_ids": list(current_removed_group),
                    "removed_time": removed_time,
                    "removed_flux": removed_flux,
                    "removed_fluxerr": removed_fluxerr,
                    "removed_snr": removed_snr,
                    "removed_band": removed_bands,
                    "removed_image": removed_images,

                    "reused_baseline": False,
                }

                eligible, reason, structure = structural_status(
                    subset,
                    config,
                )

                if not eligible:
                    records.append(
                        {
                            **metadata,
                            **structure,
                            "fit_success": False,
                            "constrained": False,
                            "failure_reason": reason,
                        }
                    )
                    break

                # Skip intermediate sizes during pilot runs.
                if n_removed % config.fit_every != 0:
                    continue

                fit_seed = make_seed(
                    config.master_seed,
                    system_number,
                    path_id,
                    n_removed + 10,
                )

                try:
                    sampler = fit_function(
                        subset,
                        fit_seed,
                    )

                    summary = summarize_delay_posterior(
                        sampler,
                        truth_delay,
                        config,
                    )

                    records.append(
                        {
                            **metadata,
                            **structure,
                            **summary,
                            "fit_success": True,
                            "failure_reason": "ok",
                        }
                    )

                    consecutive_failures = 0

                except Exception as error:
                    records.append(
                        {
                            **metadata,
                            **structure,
                            "fit_success": False,
                            "constrained": False,
                            "failure_reason": (
                                f"{type(error).__name__}: "
                                f"{error}"
                            ),
                        }
                    )

                    consecutive_failures += 1

                    if (
                        consecutive_failures
                        >= config.max_consecutive_fit_failures
                    ):
                        break

    results = pd.DataFrame.from_records(records)

    return results