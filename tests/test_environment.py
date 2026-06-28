from rubinscssl.environment import dependency_paths


def test_rubin_and_slsim_available():
    paths = dependency_paths()

    assert "/src/slsim/" in paths["slsim"]
    assert "lsst" in paths["lsst.afw.image"]
    assert "lsst" in paths["lsst.daf.butler"]
