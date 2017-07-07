"""
Test the GRS computation algorithm
"""


import unittest

import geneparse
import geneparse.testing
import numpy as np

from ..scripts import build_grs


class TestCompute(unittest.TestCase):
    def test_weight_unambiguous(self):
        # _weight_ambiguous(g, info, quality_weight):
        # Geno: reference: G / coded: T
        # Stats: risk: T
        v1 = geneparse.Variant("rs12345", 1, 123, "GT")
        g1 = geneparse.Genotypes(
            variant=v1,
            genotypes=np.array([0, 1, np.nan, 1, 0, 1, 0, 2, np.nan, 2]),
            reference="G",
            coded="T",
            multiallelic=False
        )
        info1 = build_grs.ScoreInfo(0.1, reference="G", risk="T")
        mean1 = np.nanmean(g1.genotypes)

        # Geno: reference: C / coded: A
        # Stats: risk: C
        v2 = geneparse.Variant("rs98174", 1, 456, "CA")
        g2 = geneparse.Genotypes(
            variant=v2,
            genotypes=np.array([np.nan, 2, 1, 0, 0, 2, 1, np.nan, 0, np.nan]),
            reference="C",
            coded="A",
            multiallelic=False
        )
        info2 = build_grs.ScoreInfo(0.2, reference="A", risk="C")
        mean2 = np.nanmean(g2.genotypes)

        expected = np.array([
            0 + (2 - mean2) * info2.effect,
            1 * info1.effect,
            mean1 * info1.effect + info2.effect,
            info1.effect + 2 * info2.effect,
            2 * info2.effect,
            info1.effect,
            info2.effect,
            2 * info1.effect + (2 - mean2) * info2.effect,
            mean1 * info1.effect + 2 * info2.effect,
            2 * info1.effect + (2 - mean2) * info2.effect
        ])

        grs = np.zeros(10)
        grs += build_grs._weight_unambiguous(g1, info1, False)
        grs += build_grs._weight_unambiguous(g2, info2, False)

        np.testing.assert_array_almost_equal(expected, grs)

    def test_weight_unambiguous_quality(self):
        # _weight_ambiguous(g, info, quality_weight):
        # Geno: reference: G / coded: T
        # Stats: risk: T
        v1 = geneparse.ImputedVariant("rs12345", 1, 123, "GT", quality=0.5)
        g1 = geneparse.Genotypes(
            variant=v1,
            genotypes=np.array([0, 1, np.nan, 1, 0, 1, 0, 2, np.nan, 2]),
            reference="G",
            coded="T",
            multiallelic=False
        )
        info1 = build_grs.ScoreInfo(0.1, reference="G", risk="T")
        mean1 = np.nanmean(g1.genotypes)

        # Geno: reference: C / coded: A
        # Stats: risk: C
        v2 = geneparse.ImputedVariant("rs98174", 1, 456, "CA", quality=0.8)
        g2 = geneparse.Genotypes(
            variant=v2,
            genotypes=np.array([np.nan, 2, 1, 0, 0, 2, 1, np.nan, 0, np.nan]),
            reference="C",
            coded="A",
            multiallelic=False
        )
        info2 = build_grs.ScoreInfo(0.2, reference="A", risk="C")
        mean2 = np.nanmean(g2.genotypes)

        expected = np.array([
            ((2 - mean2) * info2.effect) * 0.8,
            1 * info1.effect * 0.5,
            (mean1 * info1.effect) * 0.5 + info2.effect * 0.8,
            (info1.effect * 0.5) + (2 * info2.effect * 0.8),
            2 * info2.effect * 0.8,
            info1.effect * 0.5,
            info2.effect * 0.8,
            2 * info1.effect * 0.5 + (2 - mean2) * info2.effect * 0.8,
            mean1 * info1.effect * 0.5 + 2 * info2.effect * 0.8,
            2 * info1.effect * 0.5 + (2 - mean2) * info2.effect * 0.8
        ])

        grs = np.zeros(10)
        grs += build_grs._weight_unambiguous(g1, info1, True)
        grs += build_grs._weight_unambiguous(g2, info2, True)

        np.testing.assert_array_almost_equal(expected, grs)

    def test_weight_unambiguous_negative_effect(self):
        v1 = geneparse.Variant("testing", 1, 12345, "TC")
        g1 = geneparse.testing.simulate_genotypes_for_variant(
            v1, "T", 0.2, 1000, call_rate=0.99
        )
        info1 = build_grs.ScoreInfo(-0.2, reference="C", risk="T")

        v2 = geneparse.Variant("testing2", 2, 15161, "GA")
        g2 = geneparse.testing.simulate_genotypes_for_variant(
            v2, "G", 0.34, 1000, call_rate=0.99
        )
        info2 = build_grs.ScoreInfo(0.2, reference="G", risk="A")

        # Set the expected value for missing data.
        g1.genotypes[np.isnan(g1.genotypes)] = np.nanmean(g1.genotypes)
        g2.genotypes[np.isnan(g2.genotypes)] = np.nanmean(g2.genotypes)

        expected = (2 - g1.genotypes) * -info1.effect
        expected += (2 - g2.genotypes) * info2.effect

        observed = np.zeros(1000)
        observed += build_grs._weight_unambiguous(g1, info1, True)
        observed += build_grs._weight_unambiguous(g2, info2, True)

        np.testing.assert_array_almost_equal(expected, observed)

    def test_id_strand_frequency_noflip(self):
        my_v = geneparse.Variant("rs12345", 1, 1234151, "GC")
        my_g = geneparse.testing.simulate_genotypes_for_variant(
            my_v, "C", 0.28, 1000, call_rate=0.97
        )

        reference_g = geneparse.testing.simulate_genotypes_for_variant(
            my_v, "C", 0.28, 400, call_rate=0.99
        )

        reference = _FakeReader({my_v: reference_g})

        need_strand_flip = build_grs._id_strand_by_frequency(my_g, reference)
        self.assertFalse(need_strand_flip)

    def test_id_strand_frequency_noflip_genotypes_flipped(self):
        my_v = geneparse.Variant("rs12345", 1, 1234151, "GC")
        my_g = geneparse.testing.simulate_genotypes_for_variant(
            my_v, "C", 0.28, 1000, call_rate=0.97
        )

        reference_g = geneparse.testing.simulate_genotypes_for_variant(
            my_v, "G", 0.72, 400, call_rate=0.99
        )

        reference = _FakeReader({my_v: reference_g})

        need_strand_flip = build_grs._id_strand_by_frequency(my_g, reference)
        self.assertFalse(need_strand_flip)

    def test_id_strand_frequency_flip(self):
        my_v = geneparse.Variant("rs12345", 1, 1234151, "GC")
        my_g = geneparse.testing.simulate_genotypes_for_variant(
            my_v, "G", 0.28, 1000, call_rate=0.97
        )

        reference_g = geneparse.testing.simulate_genotypes_for_variant(
            my_v, "C", 0.28, 400, call_rate=0.99
        )

        reference = _FakeReader({my_v: reference_g})

        need_strand_flip = build_grs._id_strand_by_frequency(my_g, reference)
        self.assertTrue(need_strand_flip)

    def test_id_strand_frequency_close_50(self):
        my_v = geneparse.Variant("rs12345", 1, 1234151, "GC")
        my_g = geneparse.testing.simulate_genotypes_for_variant(
            my_v, "G", 0.5, 1000, call_rate=0.97
        )

        reference_g = geneparse.testing.simulate_genotypes_for_variant(
            my_v, "C", 0.5, 400, call_rate=0.99
        )

        reference = _FakeReader({my_v: reference_g})

        need_strand_flip = build_grs._id_strand_by_frequency(my_g, reference)
        self.assertTrue(need_strand_flip is None)

    def test_id_strand_frequency_large_freq_difference(self):
        my_v = geneparse.Variant("rs12345", 1, 1234151, "GC")
        my_g = geneparse.testing.simulate_genotypes_for_variant(
            my_v, "G", 0.01, 1000, call_rate=0.97
        )

        reference_g = geneparse.testing.simulate_genotypes_for_variant(
            my_v, "C", 0.4, 400, call_rate=0.99
        )

        reference = _FakeReader({my_v: reference_g})

        need_strand_flip = build_grs._id_strand_by_frequency(my_g, reference)
        self.assertTrue(need_strand_flip is None)


class _FakeReader(object):
    def __init__(self, d):
        self.d = d

    def get_variant_genotypes(self, v):
        return [self.d[v]]
