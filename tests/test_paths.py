import importlib
import os
from pathlib import Path

import rubinscssl.paths as paths


def test_repository_path_defaults_to_checkout(monkeypatch):
    monkeypatch.delenv("RUBINSCSSL_PATH", raising=False)
    module = importlib.reload(paths)

    expected_path = Path(module.__file__).resolve().parents[2]
    assert os.environ["RUBINSCSSL_PATH"] == str(expected_path)
    assert module.get_repository_path() == expected_path
    assert module.get_data_path() == expected_path / "data"
    assert module.get_visuals_path() == expected_path / "visuals"


def test_repository_path_honors_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RUBINSCSSL_PATH", str(tmp_path))

    assert paths.get_repository_path() == tmp_path.resolve()