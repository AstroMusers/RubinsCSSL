import pytest

from rubinscssl.environment import dependency_paths


def test_rubin_and_slsim_available():
    try:
        paths = dependency_paths()
    except ModuleNotFoundError as error:
        if error.name == "lsst" or (error.name and error.name.startswith("lsst.")):
            pytest.skip("Rubin Science Pipelines are not installed in this environment")
        raise

    assert "/src/slsim/" in paths["slsim"]
    assert "lsst" in paths["lsst.afw.image"]
    assert "lsst" in paths["lsst.daf.butler"]
