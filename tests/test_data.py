import os
from src.data import generate_injected_criteo, load_criteo_csv


def test_generate_and_load(tmp_path):
    out = tmp_path / "criteo_test.csv"
    generate_injected_criteo(str(out), nrows=100)
    assert out.exists()
    df = load_criteo_csv(str(out), nrows=100)
    assert df.shape[0] == 100
    # basic columns present
    assert 'label' in df.columns
    assert any(c.startswith('I') for c in df.columns)
    assert any(c.startswith('C') for c in df.columns)
