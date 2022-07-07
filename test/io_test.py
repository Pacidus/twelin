import sys
from os import path, remove

sys.path.append(".")

from DATA import Tools
import numpy as np


def is_like(v, vtrue, rdiff=1e-5):
    return np.mean(np.abs((v - vtrue) ** 2 / vtrue)) < rdiff


def absinf(df):
    return np.abs(df["Int2e"]) < 1e-3


def test_convert_file():
    Tools.csv2parquet("test/n_test.parquet", "test/test.csv")
    assert path.exists("test/n_test.parquet")
    Tools.parquet2csv("test/n_test.csv", "test/test.parquet")
    assert path.exists("test/n_test.csv")
    remove("test/n_test.csv")
    remove("test/n_test.parquet")


def test_stats():
    pf = Tools.pq.ParquetFile("test/test.parquet")
    stats_itt = Tools.stats(pf, sample=10**3)
    stats_dir = Tools.stats(pf, sample=10**7)
    assert (stats_itt.min == stats_dir.min).all()
    assert (stats_itt.max == stats_dir.max).all()
    assert stats_itt.N == stats_dir.N
    assert is_like(stats_itt.var, stats_dir.var, 1e-5)
    assert is_like(stats_itt.mean, stats_dir.mean, 1e-20)


def test_split():
    pf = Tools.pq.ParquetFile("test/test.parquet")

    Tools.sparq(pf, absinf, name="test/out")

    pff = Tools.pq.ParquetFile("test/out_False.parquet")
    pft = Tools.pq.ParquetFile("test/out_True.parquet")

    stats = Tools.stats(pf)
    statst = Tools.stats(pft)
    statsf = Tools.stats(pff)

    assert stats.N == statst.N + statsf.N
    assert np.abs(statst.min["Int2e"]) < 1e-3
    assert np.abs(statst.max["Int2e"]) < 1e-3
    remove("test/out_False.parquet")
    remove("test/out_True.parquet")


def test_get():
    pf = Tools.pq.ParquetFile("test/test.parquet")
    for i in range(100):
        for j in range(6):
            assert len(Tools.getrand(pf, 10**j).index) == 10**j
