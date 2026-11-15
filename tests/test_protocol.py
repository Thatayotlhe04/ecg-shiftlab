"""Protocol invariants. Run: python -m pytest -q  (needs data/processed/beats.npz)"""
import numpy as np

from shiftlab import config as C, data as D


def test_partitions_are_record_disjoint_and_complete():
    assert not set(C.DS1) & set(C.DS2)
    assert len(C.DS1) == len(C.DS2) == 22
    assert not set(C.RECORDS) & set(C.PACED)


def test_known_same_subject_pair_is_flagged():
    assert C.SUBJECT["201"] == C.SUBJECT["202"]
    assert len(set(C.SUBJECT.values())) == 43


def test_class_counts_match_the_published_partition():
    d = D.load()
    for recs, expected in ((C.DS1, [45848, 944, 3788, 414]), (C.DS2, [44240, 1837, 3220, 388])):
        got = np.bincount(d["y"][np.isin(d["rec"], recs)], minlength=4).tolist()
        assert got == expected


def test_every_beat_has_finite_features():
    d = D.load()
    assert np.isfinite(d["wave"]).all()
    assert (d["rr_pre"] > 0).all() and (d["rr_post"] > 0).all()
