import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from rubinscssl import plot_utils
from tdpy.plotting import plot_background_colors, plot_style


@pytest.fixture
def population():
    random = np.random.default_rng(42)
    indices = np.arange(120)
    return pd.DataFrame({
        'x': random.normal(size=120),
        'y': random.normal(size=120),
        'log10_abs_delay_error_plus': random.normal(size=120),
        'outcome': indices % 2 == 0,
        'curve_type': 'mock',
        'estimator': 'test',
        'band': np.tile(['g', 'r', 'i'], 40),
        'attempted': True,
        'available': True,
        'assessable': indices % 4 != 0,
        'accurate': indices % 3 != 0,
        'usable': (indices % 4 != 0) & (indices % 3 != 0),
    })


def assert_theme(figure, typeplotback, path):
    background, foreground = plot_background_colors(typeplotback)
    FigureCanvasAgg(figure).draw()
    assert figure.get_facecolor() == to_rgba(background)
    for axis in figure.axes:
        assert axis.get_facecolor() == to_rgba(background)
        assert axis.xaxis.label.get_color() == foreground
        assert axis.yaxis.label.get_color() == foreground
        assert all(spine.get_edgecolor() == to_rgba(foreground) for spine in axis.spines.values())
    for legend in figure.legends:
        assert all(text.get_color() == foreground for text in legend.get_texts())
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('RGB'))
        expected = tuple(int(channel * 255) for channel in to_rgba(background)[:3])
        assert tuple(pixels[0, 0]) == expected
        assert np.any(pixels != expected)


@pytest.mark.parametrize('kind', ['binary', 'continuous', 'bands'])
@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('inherit', [False, True])
def test_plot_apis_return_and_save_requested_theme(population, tmp_path, kind, typeplotback, inherit):
    original_data = population.copy(deep=True)
    path = tmp_path / 'figure.png'
    options = {} if inherit else {'typeplotback': typeplotback}
    outer_theme = typeplotback if inherit else ('dark' if typeplotback == 'white' else 'white')
    with plot_style(outer_theme):
        original_settings = dict(plt.rcParams)
        if kind == 'binary':
            result = plot_utils.plot_binary_corner(
                population, features=['x', 'y'], labels=['x', 'y'], outcome='outcome',
                false_label='False', true_label='True', title='Binary', filename=path, **options,
            )
            assert result['features_used'] == ['x', 'y']
            assert (result['n_false'], result['n_true']) == (60, 60)
            figure = result['figure']
        elif kind == 'continuous':
            result = plot_utils.plot_continuous_error_corner(
                population, features=['x'], labels=['x'], title='Error', filename=path, **options,
            )
            assert result['features_used'] == ['x', 'log10_abs_delay_error_plus']
            assert result['n_rows'] == len(population)
            figure = result['figure']
        else:
            summary = pd.DataFrame({
                'curve_type': ['mock'] * 3, 'estimator': ['test'] * 3, 'band': ['g', 'r', 'i'],
                'assessable_fraction_given_attempted': [0.7, 0.8, 0.9],
                'accuracy_fraction_given_assessable': [0.8, 0.9, 0.7],
                'end_to_end_accurate_fraction': [0.56, 0.72, 0.63],
            })
            figure = plot_utils.plot_band_performance(
                summary, curve_type='mock', estimator='test', filename=path, **options,
            )
            np.testing.assert_allclose([patch.get_height() for patch in figure.axes[0].patches[:3]], [0.7, 0.8, 0.9])
        try:
            assert dict(plt.rcParams) == original_settings
            assert_theme(figure, typeplotback, path)
            pd.testing.assert_frame_equal(population, original_data)
        finally:
            plt.close(figure)


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_corner_suite_forwards_theme_without_mutating_input(population, tmp_path, typeplotback):
    original_data = population.copy(deep=True)
    with plot_style('white'):
        original_settings = dict(plt.rcParams)
        outputs = plot_utils.make_population_corner_suite(
            population, curve_type='mock', estimator='test', output_directory=tmp_path,
            features=['x'], labels=['x'], bands=['g'], typeplotback=typeplotback,
        )
        assert dict(plt.rcParams) == original_settings
    assert len(outputs) == 7
    assert len(list(tmp_path.glob('*.png'))) == len(outputs)
    for result in outputs.values():
        assert 'skipped' not in result
        figure = result['figure']
        assert figure.get_facecolor() == to_rgba(plot_background_colors(typeplotback)[0])
        plt.close(figure)
    pd.testing.assert_frame_equal(population, original_data)


def test_band_performance_bokeh_uses_summary_data_and_writes_html(tmp_path, monkeypatch):
    pytest.importorskip('bokeh')
    from tdpy import plotting as tdpy_plotting

    summary = pd.DataFrame({
        'curve_type': ['observed'] * 3,
        'estimator': ['spline'] * 3,
        'band': ['g', 'r', 'i'],
        'assessable_fraction_given_attempted': [0.5, 0.6, 0.7],
        'accuracy_fraction_given_assessable': [0.8, 0.7, 0.9],
        'end_to_end_accurate_fraction': [0.4, 0.42, 0.63],
    })
    output_path = tmp_path / 'band_performance.html'
    captured = {}
    save_bokeh_html = tdpy_plotting.save_bokeh_html

    def capture_and_save(figure, path, **kwargs):
        captured['data'] = next(
            renderer.data_source.data
            for renderer in figure.renderers
            if hasattr(renderer, 'data_source')
        )
        return save_bokeh_html(figure, path, **kwargs)

    monkeypatch.setattr(tdpy_plotting, 'save_bokeh_html', capture_and_save)
    result = plot_utils.plot_band_performance_bokeh(
        summary,
        curve_type='observed',
        estimator='spline',
        filename=output_path,
    )

    assert result == output_path
    assert 'band' in captured['data']
    np.testing.assert_array_equal(captured['data']['band'], ['g', 'r', 'i'])
    np.testing.assert_allclose(
        captured['data']['end_to_end_accurate_fraction'], [0.4, 0.42, 0.63]
    )
    html = output_path.read_text(encoding='utf-8')
    assert 'Bokeh' in html
    assert '/' + 'Users/' not in html


def test_synthetic_feature_notebook_uses_pipeline_and_keeps_paths_private():
    from pathlib import Path

    path = Path(__file__).parents[1] / 'examples' / 'synthetic_feature_pipeline.ipynb'
    print('Reading from examples/synthetic_feature_pipeline.ipynb...')
    notebook = json.loads(path.read_text(encoding='utf-8'))
    code = '\n'.join(
        ''.join(cell['source'])
        for cell in notebook['cells']
        if cell['cell_type'] == 'code'
    )
    document = json.dumps(notebook)

    assert 'from rubinscssl.curve_features import build_population_analysis_tables' in code
    assert 'build_population_analysis_tables(' in code
    assert 'plot_band_performance_bokeh(' in code
    assert 'explicitly synthetic' in document
    assert '/' + 'Users/' not in document