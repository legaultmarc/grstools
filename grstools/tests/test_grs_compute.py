"""
Test the GRS computation algorithm
"""


import unittest
from pkg_resources import resource_filename

import geneparse
import geneparse.testing
import numpy as np

from ..scripts import build_grs


class TestCompute(unittest.TestCase):
    def test_weight_unambiguous(self):
        # _weight_ambiguous(g, info, quality_weight):
        # Geno: reference: G / coded: T
        # Stats: risk: T
        # *No need to flip*
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
        # *Need to flip*
        v2 = geneparse.Variant("rs98174", 1, 456, "CA")
        g2 = geneparse.Genotypes(
            variant=v2,
            # For the GRS, we will use:
            #                   NA,     0, 1, 2, 2, 0, 1, NA,     2, NA
            genotypes=np.array([np.nan, 2, 1, 0, 0, 2, 1, np.nan, 0, np.nan]),
            reference="C",
            coded="A",
            multiallelic=False
        )
        info2 = build_grs.ScoreInfo(0.2, reference="A", risk="C")
        mean2 = np.nanmean(g2.genotypes)

        assert g1.genotypes.shape[0] == g2.genotypes.shape[0]

        # When computing GRS, missing genotypes are counted as the expected
        # value of the risk allele.
        expected = np.array([
            0 + (2 - mean2) * info2.effect,
            1 * info1.effect + 0,
            mean1 * info1.effect + 1 * info2.effect,
            1 * info1.effect + 2 * info2.effect,
            0 + 2 * info2.effect,
            1 * info1.effect + 0,
            0 + info2.effect,
            2 * info1.effect + (2 - mean2) * info2.effect,
            mean1 * info1.effect + 2 * info2.effect,
            2 * info1.effect + (2 - mean2) * info2.effect
        ])

        grs = np.zeros(g1.genotypes.shape[0])
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

    def test_weight_unambiguous_bad_alleles(self):
        v1 = geneparse.Variant("testing", 1, 12345, "AG")
        g1 = geneparse.testing.simulate_genotypes_for_variant(
            v1, "A", 0.2, 1000, call_rate=0.98
        )
        info1 = build_grs.ScoreInfo(0.3, reference="T", risk="A")

        with self.assertRaises(RuntimeError):
            build_grs._weight_unambiguous(g1, info1, True)

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

    def test_replace_by_tag(self):
        reference = get_reference()

        v = geneparse.Variant("rs35391999", 2, 85626581, "AT")
        g = _FakeGenotypes(v)
        info = build_grs.ScoreInfo(0.2, reference="A", risk="T")

        g, tag_info, r2 = build_grs._replace_by_tag(
            g, info, reference, reference
        )

        reference.close()

        self.assertEqual(g.variant.name, "rs6419687")
        self.assertEqual(g.variant.chrom, 2)
        self.assertEqual(g.variant.pos, 85629817)
        self.assertEqual(g.variant.alleles_set, {"G", "A"})

        # 0.996 was computed by plink
        self.assertAlmostEqual(r2, 0.996, places=3)
        self.assertAlmostEqual(tag_info.effect, r2 * 0.2)

        self.assertEqual(tag_info.risk, "G")  # G=T, according to plink

    def test_replace_by_tag_computation_non_coded(self):
        reference = get_reference()

        v = geneparse.Variant("rs35391999", 2, 85626581, "AT")
        g = _FakeGenotypes(v)
        info = build_grs.ScoreInfo(0.2, reference="A", risk="T")

        g, tag_info, r2 = build_grs._replace_by_tag(
            g, info, reference, reference
        )

        raw_tag_geno = reference.get_variant_genotypes(g.variant)[0]

        reference.close()

        # In the file, the A alleles for both variants are coded.
        # The risk allele is T and T=G, so we need to flip the genotype
        # alleles for the expected computation.

        grs = build_grs._weight_unambiguous(g, tag_info, False)
        np.testing.assert_array_almost_equal(
            grs,
            r2 * 0.2 * (2 - raw_tag_geno.genotypes)
        )

    def test_replace_by_tag_computation_coded(self):
        reference = get_reference()

        v = geneparse.Variant("rs35391999", 2, 85626581, "AT")
        g = _FakeGenotypes(v)
        info = build_grs.ScoreInfo(0.2, reference="T", risk="A")

        g, tag_info, r2 = build_grs._replace_by_tag(
            g, info, reference, reference
        )

        raw_tag_geno = reference.get_variant_genotypes(g.variant)[0]

        reference.close()

        # In the file, the A alleles for both variants are coded.
        # The risk allele is A and A=A, so we need to use the tag's
        # genotype as is for the computation.

        grs = build_grs._weight_unambiguous(g, tag_info, False)
        np.testing.assert_array_almost_equal(
            grs,
            r2 * 0.2 * raw_tag_geno.genotypes
        )

    def test_replace_by_tag_notag(self):
        reference = get_reference()

        v = geneparse.Variant("rs12714148", 2, 85859835, "AT")
        g = _FakeGenotypes(v)
        info = build_grs.ScoreInfo(0.2, reference="A", risk="T")

        with self.assertRaises(build_grs.CouldNotFindTag):
            g, tag_info, r2 = build_grs._replace_by_tag(
                g, info, reference, reference
            )

        reference.close()


class _FakeReader(object):
    def __init__(self, d):
        self.d = d

    def get_variant_genotypes(self, v):
        return [self.d[v]]


class _FakeGenotypes(object):
    def __init__(self, variant):
        self.variant = variant


def get_reference():
    return geneparse.parsers["plink"](
        resource_filename(__name__, "data/extract_tag_test.bed")[:-4]
    )
