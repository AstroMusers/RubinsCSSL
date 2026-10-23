import importlib
import json
import pickle
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rubinscssl.curve_features import (
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


@pytest.mark.parametrize('value', [None, [], 'missing', np.nan, np.inf, -np.inf])
def test_finite_float_preserves_fallback(value):
    assert np.isnan(_finite_float(value))
    assert _finite_float(value, default=-2.0) == -2.0


@pytest.mark.parametrize('value', [2, '2', [2, 3], np.array([[2, 3]])])
def test_finite_float_returns_first_scalar(value):
    assert _finite_float(value) == 2.0


@pytest.mark.parametrize('value', [None, pd.NA, np.nan, pd.NaT])
def test_nullable_bool_retains_missing_values(value):
    assert _nullable_bool(value) is pd.NA


@pytest.mark.parametrize(('value', 'expected'), [(False, False), (0, False), (1, True), ('False', True)])
def test_nullable_bool_preserves_python_truth_values(value, expected):
    assert _nullable_bool(value) is expected


def test_curve_arrays_filter_sort_and_leave_original_arrays_unchanged():
    curve = SimpleNamespace(
        jds=np.array([4.0, 1.0, 3.0, 2.0, np.nan, 6.0, 7.0]),
        mags=np.array([20.0, 19.0, 21.0, 22.0, 20.0, np.nan, 20.0]),
        magerrs=np.array([0.2, 0.1, 0.3, 0.0, 0.1, 0.1, np.inf]),
    )
    originals = [array.copy() for array in (curve.jds, curve.mags, curve.magerrs)]

    arrays = _curve_arrays(curve)
    for actual, expected in zip(arrays, ([1.0, 3.0, 4.0], [19.0, 21.0, 20.0], [0.1, 0.3, 0.2])):
        np.testing.assert_array_equal(actual, expected)
    for actual, original in zip((curve.jds, curve.mags, curve.magerrs), originals):
        np.testing.assert_array_equal(actual, original)

    summary = _summarize_curve(curve)
    assert summary == pytest.approx({
        'n': 3,
        'first_epoch': 1.0,
        'last_epoch': 4.0,
        'baseline': 3.0,
        'median_cadence': 1.5,
        'max_gap': 2.0,
        'median_error': 0.2,
        'mean_error': 0.2,
        'median_magnitude': 20.0,
    })


def test_curve_arrays_reject_inconsistent_lengths():
    curve = SimpleNamespace(jds=[1.0], mags=[], magerrs=[0.1])
    with pytest.raises(ValueError, match='inconsistent lengths: 1, 0, 1'):
        _curve_arrays(curve)


def test_empty_curve_summary_retains_all_columns():
    summary = _summarize_curve(SimpleNamespace(jds=[], mags=[], magerrs=[]))
    assert summary.pop('n') == 0
    assert set(summary) == {
        'first_epoch', 'last_epoch', 'baseline', 'median_cadence',
        'max_gap', 'median_error', 'mean_error', 'median_magnitude',
    }
    assert all(np.isnan(value) for value in summary.values())


def test_single_epoch_curve_summary_retains_zero_baseline():
    summary = _summarize_curve(SimpleNamespace(jds=[2.0], mags=[20.0], magerrs=[0.1]))
    assert summary['n'] == 1
    assert summary['first_epoch'] == summary['last_epoch'] == 2.0
    assert summary['baseline'] == 0.0
    assert np.isnan(summary['median_cadence'])
    assert np.isnan(summary['max_gap'])


def test_feature_table_preserves_curve_types_full_baselines_and_missing_rows():
    first_curve = SimpleNamespace(jds=[4.0, 1.0, 3.0], mags=[20.0, 19.0, 21.0], magerrs=[0.2, 0.1, 0.3])
    second_curve = SimpleNamespace(jds=[2.0, 3.0, 5.0], mags=[22.0, 21.0, 23.0], magerrs=[0.2, 0.2, 0.2])
    systems = {
        'valid': {
            'system_status': 'success', 'time_delay': [1.0, 3.0],
            'z_lens': [0.5], 'z_source': 1.0, 'einstein_radius': 2.0,
            'r': {'band_status': 'success', 'model_lcs': [first_curve, second_curve], 'lcs': [first_curve, second_curve]},
        },
        'failed': {'system_status': 'failed'},
    }

    table = build_light_curve_feature_table_from_lcs(systems, bands=['r', 'g'])

    assert len(table) == 8
    assert not table.duplicated(['system_id', 'band', 'curve_type']).any()
    successful = table[table['feature_status'].eq('success')]
    assert set(successful['curve_type']) == {'model', 'observed'}
    for _, row in successful.iterrows():
        for column, expected in {
            'n_A': 3, 'n_B': 3, 'n_min': 3, 'n_max': 3, 'n_total': 6,
            'n_shared': 1, 'shared_fraction': 1 / 3,
            'baseline_A': 3.0, 'baseline_B': 3.0, 'baseline_min': 3.0,
            'overlap_baseline': 2.0, 'median_cadence': 1.5, 'max_gap': 2.0,
            'median_error': 0.2, 'median_image_contrast': 2.0,
            'points_per_100_days': 150.0, 'delay_to_overlap_ratio': 1.0,
            'true_delay_feature': 2.0, 'z_lens': 0.5,
        }.items():
            assert row[column] == pytest.approx(expected)
    assert table['feature_status'].value_counts().to_dict() == {
        'success': 2, 'band_unavailable': 2, 'system_unavailable': 4,
    }


@pytest.mark.parametrize(
    ('curves', 'status'),
    [
        ([SimpleNamespace(jds=[], mags=[], magerrs=[])], 'invalid_curve_pair'),
        ([SimpleNamespace(jds=[1.0], mags=[], magerrs=[0.1])] * 2, 'feature_extraction_failed'),
    ],
)
def test_feature_table_retains_invalid_pairs_and_extraction_errors(curves, status):
    systems = {'failed': {'system_status': 'success', 'r': {
        'band_status': 'success', 'model_lcs': curves, 'lcs': curves,
    }}}

    table = build_light_curve_feature_table_from_lcs(systems, bands=['r'])

    assert len(table) == 2
    assert table['feature_status'].tolist() == [status, status]
    if status == 'feature_extraction_failed':
        assert table['feature_exception_type'].tolist() == ['ValueError', 'ValueError']


def test_feature_table_rejects_repeated_bands_and_preserves_empty_input():
    assert build_light_curve_feature_table_from_lcs({}).empty
    with pytest.raises(ValueError, match='not unique'):
        build_light_curve_feature_table_from_lcs({'failed': {}}, bands=['r', 'r'])


@pytest.mark.parametrize('module_name', ['pycs3_utils', 'pycs3_utils_multistart_parallel'])
def test_fitting_modules_reexport_the_shared_curve_helpers(module_name):
    pytest.importorskip('pycs3')
    module = importlib.import_module(f'rubinscssl.{module_name}')
    for helper in (
        _finite_float, _nullable_bool, _curve_arrays, _summarize_curve,
        build_light_curve_feature_table_from_lcs,
        flatten_baseline_results, add_population_derived_columns,
        build_population_analysis_tables, summarize_band_performance,
        binary_feature_correlations, continuous_error_correlations,
        classify_mock_measurement_success, add_estimator_agreement,
        _as_float, _flatten_mock_row, build_mock_measurement_table,
        save_successful_time_delay_systems,
    ):
        assert getattr(module, helper.__name__) is helper


def test_feature_table_accepts_real_pycs3_curves():
    pytest.importorskip('pycs3')
    from pycs3.gen.lc_func import factory

    curve = factory(np.array([1.0, 2.0, 3.0]), np.array([20.0, 21.0, 20.0]), np.array([0.1, 0.1, 0.1]))
    systems = {'valid': {'system_status': 'success', 'r': {
        'band_status': 'success', 'model_lcs': [curve, curve], 'lcs': [curve, curve],
    }}}

    table = build_light_curve_feature_table_from_lcs(systems, bands=['r'])

    assert table['feature_status'].tolist() == ['success', 'success']
    assert table['n_shared'].tolist() == [3, 3]
    assert table['baseline_min'].tolist() == [2.0, 2.0]


@pytest.mark.parametrize('module_name', ['pycs3_utils', 'pycs3_utils_multistart_parallel'])
def test_population_tables_retain_all_failed_systems_without_feature_columns(module_name):
    pytest.importorskip('pycs3')
    module = importlib.import_module(f'rubinscssl.{module_name}')
    systems = {'failed': {'system_status': 'failed'}}
    results = {'spline': [{'system_id': 'failed', 'band': 'r', 'true_delay': 2.0}]}

    tables = module.build_population_analysis_tables(systems, results)

    analysis = tables['analysis']
    assert len(analysis) == 2
    assert analysis['feature_status'].tolist() == ['system_unavailable'] * 2
    assert not analysis['available'].any()
    assert not analysis['usable'].any()
    assert analysis[['n_min', 'overlap_baseline', 'median_cadence', 'max_gap', 'median_error']].isna().all().all()
    summary = tables['band_summary']
    assert summary['n_rows'].tolist() == [1, 1]
    assert summary['n_available'].tolist() == [0, 0]
    assert summary['n_accurate'].tolist() == [0, 0]


@pytest.mark.parametrize('systems', [{}, {'failed': {'system_status': 'failed'}}])
@pytest.mark.parametrize('results', [{}, {'spline': []}, {'spline': [], 'regdiff': []}])
def test_population_tables_accept_empty_estimator_workloads(systems, results):
    tables = build_population_analysis_tables(systems, results)

    assert set(tables) == {'results', 'features', 'analysis', 'band_summary'}
    assert tables['results'].empty
    assert tables['analysis'].empty
    assert tables['band_summary'].empty
    assert {'system_id', 'band', 'curve_type', 'estimator', 'accurate'} <= set(tables['results'])
    assert {'available', 'usable', 'n_min', 'log10_n_min_plus1'} <= set(tables['analysis'])
    assert {'n_available', 'n_accurate', 'rmse_delay'} <= set(tables['band_summary'])
    assert len(tables['features']) == 6 * len(systems)


def test_empty_result_and_summary_tables_keep_nonempty_column_order():
    nonempty_results = flatten_baseline_results([{'system_id': 'failed', 'band': 'r'}], estimator='spline')
    empty_results = flatten_baseline_results([], estimator='spline')
    assert empty_results.columns.tolist() == nonempty_results.columns.tolist()
    nonempty = build_population_analysis_tables(
        {'failed': {'system_status': 'failed'}},
        {'spline': [{'system_id': 'failed', 'band': 'r'}]},
    )['band_summary']
    assert summarize_band_performance(pd.DataFrame()).columns.tolist() == nonempty.columns.tolist()


def test_mixed_population_preserves_counts_and_conditioned_success_fractions():
    curve = SimpleNamespace(jds=[1.0, 2.0, 3.0], mags=[20.0] * 3, magerrs=[0.1] * 3)
    valid_system = {
        'system_status': 'success', 'time_delay': [0.0, 2.0],
        'r': {'band_status': 'success', 'lcs': [curve, curve], 'model_lcs': [curve, curve]},
    }
    systems = {
        'fitted': valid_system,
        'unattempted': valid_system,
        'failed': {'system_status': 'failed'},
    }
    fitted = {
        'system_id': 'fitted', 'band': 'r', 'true_delay': 2.0,
        'model_spline_results': {'converged': True, 'delay': 2.5},
        'model_spline_assessment': {'assessable': True, 'accurate': True},
        'observed_spline_results': {'converged': True, 'delay': 5.0},
        'observed_spline_assessment': {'assessable': True, 'accurate': False},
    }
    results = {'spline': [
        fitted,
        {'system_id': 'unattempted', 'band': 'r'},
        {'system_id': 'failed', 'band': 'r'},
        {'system_id': 'fitted', 'band': 'g'},
    ]}

    tables = build_population_analysis_tables(systems, results)

    assert len(tables['analysis']) == 8
    summary = tables['band_summary'].set_index(['curve_type', 'band'])
    for curve_type, accurate, error in [('model', 1, 0.5), ('observed', 0, 3.0)]:
        row = summary.loc[(curve_type, 'r')]
        assert row['n_rows'] == 3
        assert row['n_available'] == 2
        assert row['n_attempted'] == row['n_assessable'] == 1
        assert row['n_accurate'] == accurate
        assert row['attempted_fraction_given_available'] == 0.5
        assert row['assessable_fraction_given_attempted'] == 1.0
        assert row['accuracy_fraction_given_assessable'] == accurate
        assert row['end_to_end_accurate_fraction'] == accurate / 2
        assert row['median_abs_delay_error'] == error
        assert row['rmse_delay'] == error
        assert summary.loc[(curve_type, 'g'), 'n_available'] == 0


def test_derived_columns_do_not_mutate_the_input_analysis_table():
    original = pd.DataFrame({
        'feature_status': ['success'], 'accurate': [True], 'assessable': [True],
        'n_min': ['3'], 'overlap_baseline': ['2'], 'median_cadence': ['1'],
        'max_gap': ['1'], 'median_error': ['0.1'], 'true_delay': ['2'],
        'abs_delay_error': ['0.5'],
    })
    snapshot = original.copy(deep=True)

    derived = add_population_derived_columns(original)

    pd.testing.assert_frame_equal(original, snapshot)
    assert derived.loc[0, 'log10_n_min_plus1'] == pytest.approx(np.log10(4.0))
    assert derived.loc[0, 'log10_abs_delay_error_plus'] == pytest.approx(np.log10(0.5))


@pytest.mark.parametrize('module_name', ['pycs3_utils', 'pycs3_utils_multistart_parallel'])
@pytest.mark.parametrize('kind', ['binary', 'continuous'])
@pytest.mark.parametrize('constant', ['feature', 'target'])
def test_feature_correlations_skip_constant_inputs_without_warnings(module_name, kind, constant):
    pytest.importorskip('pycs3')
    module = importlib.import_module(f'rubinscssl.{module_name}')
    target = 'outcome' if kind == 'binary' else 'abs_delay_error'
    frame = pd.DataFrame({
        'feature': [1.0, 2.0, 3.0, 4.0],
        'outcome': [False, True, False, True],
        'abs_delay_error': [1.0, 3.0, 2.0, 4.0],
    })
    frame['feature' if constant == 'feature' else target] = 1.0
    function = module.binary_feature_correlations if kind == 'binary' else module.continuous_error_correlations
    kwargs = {'outcome': 'outcome'} if kind == 'binary' else {}

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        table = function(frame, features=['feature'], **kwargs)

    assert table.empty
    assert {'feature', 'n', 'p_value_naive'} <= set(table.columns)


@pytest.mark.parametrize('kind', ['binary', 'continuous'])
@pytest.mark.parametrize('module_name', ['pycs3_utils', 'pycs3_utils_multistart_parallel'])
def test_shared_correlations_match_scipy_for_valid_pairs_and_absolute_sorting(kind, module_name):
    from scipy.stats import pointbiserialr, spearmanr

    pytest.importorskip('pycs3')
    module = importlib.import_module(f'rubinscssl.{module_name}')
    frame = pd.DataFrame({
        'linear': [1.0, 2.0, 3.0, 4.0, 5.0, np.inf],
        'inverse': [5.0, 4.0, 3.0, 2.0, 1.0, np.nan],
        'mixed': [4.0, 1.0, 3.0, 2.0, 5.0, 10.0],
        'outcome': [False, False, True, True, True, True],
        'abs_delay_error': [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
    })
    features = ['mixed', 'linear', 'inverse']
    target = 'outcome' if kind == 'binary' else 'abs_delay_error'
    correlation_column = 'correlation' if kind == 'binary' else 'spearman_r'
    if kind == 'binary':
        table = module.binary_feature_correlations(frame, features=features, outcome=target)
    else:
        table = module.continuous_error_correlations(frame, features=features)

    assert len(table) == 3
    magnitudes = table[correlation_column].abs().to_numpy()
    assert np.all(magnitudes[:-1] >= magnitudes[1:])
    for _, row in table.iterrows():
        pairs = frame[[row['feature'], target]].replace([np.inf, -np.inf], np.nan).dropna()
        if kind == 'binary':
            expected = pointbiserialr(pairs[target].astype(int), pairs[row['feature']].astype(float))
        else:
            expected = spearmanr(pairs[row['feature']].astype(float), pairs[target].astype(float))
        assert row['n'] == len(pairs)
        assert row[correlation_column] == pytest.approx(expected.statistic)
        assert row['p_value_naive'] == pytest.approx(expected.pvalue)


@pytest.mark.parametrize('module_name', ['pycs3_utils', 'pycs3_utils_multistart_parallel'])
@pytest.mark.parametrize('empty', [False, True])
def test_mock_classifier_retains_sparse_inputs_without_selecting_them(module_name, empty):
    pytest.importorskip('pycs3')
    module = importlib.import_module(f'rubinscssl.{module_name}')
    table = pd.DataFrame() if empty else pd.DataFrame({'mock_status': ['not_run']}, index=['pending'])
    original = table.copy(deep=True)

    classified = module.classify_mock_measurement_success(table)

    pd.testing.assert_frame_equal(table, original)
    pd.testing.assert_index_equal(classified.index, table.index)
    for column in ('mock_calibrated', 'measured_successfully', 'constrained', 'validated_success'):
        assert classified[column].dtype == bool
        assert not classified[column].any()
    assert classified.attrs['selection_criteria']['minimum_finite_mocks'] == 400
    if not empty:
        assert classified['selection_reason'].tolist() == ['observed_delay_not_assessable']


@pytest.fixture
def mock_measurement():
    return {
        'system_id': 'lens', 'band': 'r', 'estimator': 'spline',
        'mock_status': 'success', 'production_status': 'success',
        'production_calibrated': True, 'production_n_finite': 500,
        'observed_mock_failure_fraction': 0.05,
        'observed_mock_catastrophic_fraction': 0.05,
        'observed_mock_total_error': 2.0, 'observed_delay_raw': 20.0,
        'relative_uncertainty_measured': 0.1, 'delay_significance': 10.0,
        'assessable': True, 'accurate': True, 'truth_residual_raw': 1.0,
    }


@pytest.mark.parametrize(
    ('column', 'value', 'reason'),
    [
        ('mock_status', 'failed', 'production_mock_calibration_failed'),
        ('production_status', 'failed', 'production_mock_calibration_failed'),
        ('production_calibrated', False, 'production_mock_calibration_failed'),
        ('production_n_finite', 399, 'production_mock_calibration_failed'),
        ('observed_mock_failure_fraction', 0.21, 'production_mock_calibration_failed'),
        ('observed_mock_catastrophic_fraction', 0.21, 'production_mock_calibration_failed'),
        ('relative_uncertainty_measured', 0.21, 'relative_uncertainty_too_large'),
        ('delay_significance', 2.9, 'delay_not_significantly_nonzero'),
        ('accurate', False, 'inaccurate_against_simulation_truth'),
        ('truth_residual_raw', 4.1, 'truth_outside_calibrated_interval'),
        ('assessable', False, 'observed_delay_not_assessable'),
        ('observed_delay_raw', np.nan, 'observed_delay_not_assessable'),
    ],
)
def test_mock_classifier_preserves_each_rejection_condition(mock_measurement, column, value, reason):
    mock_measurement[column] = value
    table = pd.DataFrame([mock_measurement], index=['lens-row'])

    classified = classify_mock_measurement_success(table)

    assert not classified.loc['lens-row', 'validated_success']
    assert classified.loc['lens-row', 'selection_reason'] == reason


@pytest.mark.parametrize('missing', [
    'production_n_finite', 'observed_mock_failure_fraction',
    'observed_mock_catastrophic_fraction', 'observed_mock_total_error',
    'observed_delay_raw', 'relative_uncertainty_measured', 'delay_significance',
    'truth_residual_raw', 'assessable', 'accurate', 'mock_status',
])
def test_mock_classifier_does_not_select_rows_with_missing_required_evidence(mock_measurement, missing):
    del mock_measurement[missing]
    table = pd.DataFrame([mock_measurement], index=['missing-evidence'])

    classified = classify_mock_measurement_success(table)

    assert not classified.loc['missing-evidence', 'validated_success']


def test_mock_classifier_selects_complete_rows_and_preserves_input(mock_measurement):
    table = pd.DataFrame([mock_measurement], index=['complete'])
    original = table.copy(deep=True)

    classified = classify_mock_measurement_success(table)

    pd.testing.assert_frame_equal(table, original)
    assert classified.loc['complete', 'mock_calibrated']
    assert classified.loc['complete', 'constrained']
    assert classified.loc['complete', 'validated_success']
    assert classified.loc['complete', 'selection_reason'] == 'selected'


def test_mock_classifier_accepts_numeric_strings_and_inclusive_boundaries(mock_measurement):
    mock_measurement.update(
        production_n_finite='400', observed_mock_failure_fraction='0.20',
        observed_mock_catastrophic_fraction='0.20', observed_mock_total_error='2',
        observed_delay_raw='10', relative_uncertainty_measured='0.20',
        delay_significance='5', truth_residual_raw='4',
    )

    classified = classify_mock_measurement_success(pd.DataFrame([mock_measurement]))

    assert classified.loc[0, 'validated_success']


def test_mock_classifier_allows_explicit_opt_out_of_truth_requirements(mock_measurement):
    del mock_measurement['accurate']
    del mock_measurement['truth_residual_raw']

    classified = classify_mock_measurement_success(
        pd.DataFrame([mock_measurement]),
        require_baseline_accuracy=False, require_truth_coverage=False,
    )

    assert classified.loc[0, 'validated_success']
    assert not classified.attrs['selection_criteria']['require_truth_coverage']


@pytest.mark.parametrize('maximum_sigma', [0.5, 2.0])
def test_estimator_agreement_preserves_pairs_singletons_and_unconstrained_rows(maximum_sigma):
    table = pd.DataFrame({
        'system_id': ['pair', 'pair', 'single', 'unconstrained'],
        'band': ['r'] * 4, 'estimator': ['spline', 'regdiff', 'spline', 'spline'],
        'constrained': [True, True, True, False],
        'observed_delay_raw': [20.0, 24.0, 22.0, 30.0],
        'observed_mock_total_error': [3.0, 4.0, 2.0, 1.0],
    })
    original = table.copy(deep=True)

    agreement = add_estimator_agreement(table, maximum_disagreement_sigma=maximum_sigma)

    pd.testing.assert_frame_equal(table, original)
    assert len(agreement) == len(table)
    pair = agreement[agreement['system_id'].eq('pair')]
    np.testing.assert_allclose(pair['estimator_disagreement_sigma'], 0.8)
    assert pair['estimators_agree'].tolist() == [0.8 <= maximum_sigma] * 2
    assert agreement.loc[agreement['system_id'].ne('pair'), 'estimator_disagreement_sigma'].isna().all()


@pytest.mark.parametrize('classified', [False, True])
def test_empty_mock_selection_flows_through_estimator_agreement(classified):
    table = pd.DataFrame()
    if classified:
        table = classify_mock_measurement_success(table)
    original = table.copy(deep=True)

    agreement = add_estimator_agreement(table)

    pd.testing.assert_frame_equal(table, original)
    assert agreement.empty
    assert agreement['estimator_disagreement_sigma'].dtype == float
    assert agreement['estimators_agree'].dtype == bool
    assert set(table.columns) <= set(agreement.columns)


@pytest.mark.parametrize('module_name', ['pycs3_utils', 'pycs3_utils_multistart_parallel'])
def test_mock_measurements_without_known_truth_keep_measured_uncertainties(module_name):
    pytest.importorskip('pycs3')
    module = importlib.import_module(f'rubinscssl.{module_name}')
    analysis = pd.DataFrame([{
        'system_id': 'observed', 'band': 'r', 'estimator': 'spline',
        'curve_type': 'observed', 'estimated_delay': 20.0, 'assessable': True,
    }])
    original = analysis.copy(deep=True)
    mocks = {'spline': [{
        'system_id': 'observed', 'band': 'r', 'mock_status': 'success',
        'observed_delay_raw': 20.0, 'observed_mock_total_error': 2.0,
    }]}

    measured = module.build_mock_measurement_table(analysis, mocks)

    pd.testing.assert_frame_equal(analysis, original)
    assert measured.loc[0, 'relative_uncertainty_measured'] == 0.1
    assert measured.loc[0, 'delay_significance'] == 10.0
    assert measured.loc[0, 'ci68_low'] == 18.0
    assert measured.loc[0, 'ci68_high'] == 22.0
    assert measured[['relative_uncertainty_true', 'truth_residual_raw', 'truth_pull_raw']].isna().all().all()
    assert not measured[['truth_covered_68', 'truth_covered_95']].any().any()


@pytest.fixture
def completed_mock_run():
    return {
        'system_id': 'observed', 'band': 'r', 'mock_status': 'success',
        'diagnostic_result': {'status': 'success', 'n_finite': 40},
        'production_result': {
            'status': 'success', 'calibrated': True, 'n_finite': 500,
            'median_bias': 1.0, 'random_error': 1.5, 'total_error': 2.0,
            'failure_fraction': 0.1, 'catastrophic_fraction': 0.05,
        },
    }


@pytest.mark.parametrize('value', [None, 'missing', np.nan, np.inf, -np.inf, [], [1, 2]])
def test_scalar_float_conversion_keeps_fallback_semantics(value):
    assert np.isnan(_as_float(value))
    assert _as_float(value, default=-2.0) == -2.0


@pytest.mark.parametrize('value', [2, '2', np.float64(2), np.array(2)])
def test_scalar_float_conversion_accepts_finite_scalar_values(value):
    assert _as_float(value) == 2.0


@pytest.mark.parametrize('summary_bias', [None, 0.5])
def test_flatten_mock_row_prefers_finite_summary_then_production_fallback(completed_mock_run, summary_bias):
    completed_mock_run['observed_mock_bias'] = summary_bias

    flattened = _flatten_mock_row(completed_mock_run, 'spline')

    assert flattened['observed_mock_bias'] == (1.0 if summary_bias is None else summary_bias)
    assert flattened['observed_mock_random_error'] == 1.5
    assert flattened['observed_mock_total_error'] == 2.0
    assert flattened['observed_mock_failure_fraction'] == 0.1
    assert flattened['observed_mock_catastrophic_fraction'] == 0.05
    assert flattened['production_n_finite'] == 500
    assert flattened['diagnostic_n_finite'] == 40
    assert flattened['estimator'] == 'spline'


@pytest.mark.parametrize('raw_delay', [None, 25.0])
def test_mock_measurement_bias_precision_and_intervals_preserve_baseline_fallback(completed_mock_run, raw_delay):
    completed_mock_run['observed_delay_raw'] = raw_delay
    analysis = pd.DataFrame([{
        'system_id': 'observed', 'band': 'r', 'estimator': 'spline',
        'curve_type': 'observed', 'estimated_delay': 20.0,
        'assessable': True, 'accurate': True, 'true_delay': 22.0,
    }])
    expected_delay = 20.0 if raw_delay is None else raw_delay

    measured = build_mock_measurement_table(analysis, {'spline': [completed_mock_run]})

    row = measured.iloc[0]
    assert row['observed_delay_raw'] == expected_delay
    assert row['observed_delay_bias_corrected'] == expected_delay - 1.0
    assert row['relative_uncertainty_measured'] == pytest.approx(2.0 / expected_delay)
    assert row['relative_uncertainty_true'] == pytest.approx(2.0 / 22.0)
    assert row['delay_significance'] == expected_delay / 2.0
    assert row['truth_residual_raw'] == expected_delay - 22.0
    assert row['truth_pull_raw'] == (expected_delay - 22.0) / 2.0
    for label, width in [('68', 2.0), ('95', 3.92)]:
        assert row[f'ci{label}_low'] == pytest.approx(expected_delay - width)
        assert row[f'ci{label}_high'] == pytest.approx(expected_delay + width)
        assert bool(row[f'truth_covered_{label}']) == (abs(expected_delay - 22.0) <= width)


def test_observed_only_measurement_can_use_truth_independent_selection(completed_mock_run):
    analysis = pd.DataFrame([{
        'system_id': 'observed', 'band': 'r', 'estimator': 'spline',
        'curve_type': 'observed', 'estimated_delay': 20.0, 'assessable': True,
    }])

    measured = build_mock_measurement_table(analysis, {'spline': [completed_mock_run]})
    simulation_selection = classify_mock_measurement_success(measured)
    observed_selection = classify_mock_measurement_success(
        measured, require_baseline_accuracy=False, require_truth_coverage=False,
    )
    agreement = add_estimator_agreement(observed_selection)

    assert not simulation_selection.loc[0, 'validated_success']
    assert observed_selection.loc[0, 'constrained']
    assert observed_selection.loc[0, 'validated_success']
    assert observed_selection.loc[0, 'selection_reason'] == 'selected'
    assert np.isnan(agreement.loc[0, 'estimator_disagreement_sigma'])


@pytest.mark.parametrize('mock_runs', [{}, {'spline': []}])
def test_empty_mock_construction_flows_through_selection_and_agreement(mock_runs):
    measured = build_mock_measurement_table(pd.DataFrame(), mock_runs)
    classified = classify_mock_measurement_success(measured)
    agreement = add_estimator_agreement(classified)

    assert measured.empty
    assert classified.empty
    assert agreement.empty
    assert 'estimators_agree' in agreement


def test_mock_measurements_reject_duplicate_system_band_estimator_keys(completed_mock_run):
    analysis = pd.DataFrame([{
        'system_id': 'observed', 'band': 'r', 'estimator': 'spline',
        'curve_type': 'observed', 'true_delay': 22.0,
    }])
    with pytest.raises(pd.errors.MergeError):
        build_mock_measurement_table(analysis, {'spline': [completed_mock_run] * 2})


@pytest.mark.parametrize('module_name', ['pycs3_utils', 'pycs3_utils_multistart_parallel'])
def test_successful_system_export_round_trips_all_artifact_contents(module_name, tmp_path):
    pytest.importorskip('pycs3')
    module = importlib.import_module(f'rubinscssl.{module_name}')
    systems = {1: {'redshift': 0.5, 'samples': np.array([1.0, 2.0])}, 2: {'redshift': 1.0}}
    selection = pd.DataFrame({
        'system_id': [1, 2], 'band': ['r', 'g'], 'estimator': ['spline', 'regdiff'],
        'validated_success': [True, False], 'coords': [(1.0, 2.0), (3.0, 4.0)],
    })
    selection.attrs['selection_criteria'] = {'minimum_finite_mocks': 400}
    original = selection.copy(deep=True)

    paths = module.save_successful_time_delay_systems(systems, selection, tmp_path)

    assert set(paths) == {
        'measurements_pickle', 'measurements_csv', 'systems_pickle',
        'system_ids_pickle', 'criteria_json', 'band_r', 'estimator_spline',
    }
    print(f"Reading from {paths['measurements_pickle']}...")
    measurements = pd.read_pickle(paths['measurements_pickle'])
    pd.testing.assert_frame_equal(measurements, selection.iloc[[0]].reset_index(drop=True))
    print(f"Reading from {paths['measurements_csv']}...")
    csv = pd.read_csv(paths['measurements_csv'])
    pd.testing.assert_frame_equal(csv, measurements.drop(columns=['coords']))
    for key in ('band_r', 'estimator_spline'):
        print(f'Reading from {paths[key]}...')
        pd.testing.assert_frame_equal(pd.read_pickle(paths[key]), measurements)
    print(f"Reading from {paths['systems_pickle']}...")
    with paths['systems_pickle'].open('rb') as stream:
        restored = pickle.load(stream)
    assert set(restored) == {1}
    assert restored[1]['redshift'] == 0.5
    np.testing.assert_array_equal(restored[1]['samples'], systems[1]['samples'])
    assert restored[1]['successful_time_delay_measurements'] == [{
        'system_id': 1, 'band': 'r', 'estimator': 'spline', 'validated_success': True,
    }]
    print(f"Reading from {paths['system_ids_pickle']}...")
    with paths['system_ids_pickle'].open('rb') as stream:
        assert pickle.load(stream) == [1]
    print(f"Reading from {paths['criteria_json']}...")
    criteria = json.loads(paths['criteria_json'].read_text())
    assert criteria == {
        'minimum_finite_mocks': 400, 'selection_column': 'validated_success',
        'n_selected_measurements': 1, 'n_selected_systems': 1,
    }
    pd.testing.assert_frame_equal(selection, original)
    assert 'successful_time_delay_measurements' not in systems[1]


@pytest.mark.parametrize('per_band', [False, True])
@pytest.mark.parametrize('per_estimator', [False, True])
def test_export_options_and_narration_match_the_returned_artifacts(tmp_path, capsys, per_band, per_estimator):
    selection = pd.DataFrame({
        'system_id': [1], 'band': ['r'], 'estimator': ['spline'],
        'validated_success': [True],
    })

    paths = save_successful_time_delay_systems(
        {1: {'redshift': 0.5}}, selection, tmp_path, file_stem='chosen',
        save_per_band=per_band, save_per_estimator=per_estimator,
    )

    assert capsys.readouterr().out.splitlines() == [f'Writing to {path}...' for path in paths.values()]
    assert ('band_r' in paths) == per_band
    assert ('estimator_spline' in paths) == per_estimator
    assert all(path.name.startswith('chosen_') for path in paths.values())
    assert set(tmp_path.iterdir()) == set(paths.values())


def test_empty_classification_exports_empty_tables_catalog_ids_and_counts(tmp_path):
    selection = classify_mock_measurement_success(pd.DataFrame())

    paths = save_successful_time_delay_systems({}, selection, tmp_path)

    assert len(paths) == 5
    print(f"Reading from {paths['measurements_pickle']}...")
    measurements = pd.read_pickle(paths['measurements_pickle'])
    assert measurements.empty
    assert {'system_id', 'band', 'estimator', 'validated_success'} <= set(measurements)
    print(f"Reading from {paths['measurements_csv']}...")
    csv = pd.read_csv(paths['measurements_csv'])
    assert csv.empty
    assert csv.columns.tolist() == measurements.columns.tolist()
    for key, expected in [('systems_pickle', {}), ('system_ids_pickle', [])]:
        print(f'Reading from {paths[key]}...')
        with paths[key].open('rb') as stream:
            assert pickle.load(stream) == expected
    print(f"Reading from {paths['criteria_json']}...")
    criteria = json.loads(paths['criteria_json'].read_text())
    assert criteria['n_selected_measurements'] == 0
    assert criteria['n_selected_systems'] == 0
    assert criteria['minimum_finite_mocks'] == 400


def test_export_preserves_custom_selection_column_and_nullable_flags(tmp_path):
    selection = pd.DataFrame({
        'system_id': [1, 2, 3], 'band': ['r', 'r', 'r'],
        'estimator': ['spline', 'spline', 'spline'],
        'keep': pd.array([True, False, pd.NA], dtype='boolean'),
    })

    paths = save_successful_time_delay_systems(
        {1: {'redshift': 0.5}, 2: {}, 3: {}}, selection, tmp_path,
        selection_column='keep', save_per_band=False, save_per_estimator=False,
    )

    print(f"Reading from {paths['measurements_pickle']}...")
    assert pd.read_pickle(paths['measurements_pickle'])['system_id'].tolist() == [1]
    print(f"Reading from {paths['criteria_json']}...")
    assert json.loads(paths['criteria_json'].read_text())['selection_column'] == 'keep'


def test_export_rejects_missing_selection_before_creating_outputs(tmp_path):
    with pytest.raises(KeyError, match='Missing selection column'):
        save_successful_time_delay_systems({}, pd.DataFrame(), tmp_path / 'unused')
    assert list(tmp_path.iterdir()) == []
