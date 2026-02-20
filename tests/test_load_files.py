import numpy as np
import pandas as pd

from src.Miscellaneous import load_file


SAMPLE_DATA = np.array([[2.0, 1.0, -100.0], [3.0, 2.0, -120.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def test_load_csv(tmp_path):
    path = str(tmp_path / "data.csv")
    pd.DataFrame(SAMPLE_DATA).to_csv(path, index=False, header=False)
    loaded = load_file(path)
    assert loaded is not None
    np.testing.assert_array_almost_equal(loaded, SAMPLE_DATA)


def test_load_txt(tmp_path):
    path = str(tmp_path / "data.txt")
    np.savetxt(path, SAMPLE_DATA)
    loaded = load_file(path)
    assert loaded is not None
    np.testing.assert_array_almost_equal(loaded, SAMPLE_DATA)


def test_load_xlsx(tmp_path):
    path = str(tmp_path / "data.xlsx")
    pd.DataFrame(SAMPLE_DATA).to_excel(path, index=False, header=False)
    loaded = load_file(path)
    assert loaded is not None
    np.testing.assert_array_almost_equal(loaded, SAMPLE_DATA)


def test_load_json(tmp_path):
    path = str(tmp_path / "data.json")
    pd.DataFrame(SAMPLE_DATA).to_json(path)
    loaded = load_file(path)
    assert loaded is not None
    np.testing.assert_array_almost_equal(loaded, SAMPLE_DATA)


def test_load_unsupported_format(tmp_path):
    """Unsupported file format returns None."""
    path = str(tmp_path / "data.yaml")
    with open(path, "w") as f:
        f.write("key: value\n")
    result = load_file(path)
    assert result is None


def test_load_missing_file():
    """Non-existent file returns None."""
    result = load_file("/tmp/nonexistent_file_12345.csv")
    assert result is None
