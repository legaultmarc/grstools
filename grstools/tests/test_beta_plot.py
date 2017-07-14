#!/usr/bin/env python
"""
Test the GRS utils beta-plot algorithm
"""


import unittest
from multiprocessing import cpu_count
from pkg_resources import resource_filename

import pandas as pd

import geneparse
from ..scripts.utils.beta_plot import BetaTuple
from ..scripts.utils.beta_plot import BetaSubscriber
from ..scripts.utils.beta_plot import compute_beta_coefficients


class TestCompute(unittest.TestCase):
    def test_BetaSubscriber_handle_noAlleleFlip(self):
        v1 = geneparse.Variant(None, 2, 12345, ["A", "G"])
        b1 = BetaTuple("A", 0.45)
        v2 = geneparse.Variant(None, 21, 6789, ["C", "T"])
        b2 = BetaTuple("T", -0.91)

        variant_to_expected = {v1: b1, v2: b2}

        result1 = {'MODEL': {'nobs': 3767},
                   'SNPs': {'chrom': 'chr2',
                            'pos': 12345,
                            'coef': 0.45,
                            'major': 'G',
                            'minor': 'A',
                            'std_err': 0.07027076759612147,
                            'maf': 0.060788425803026284}
                   }

        result2 = {'MODEL': {'nobs': 2021},
                   'SNPs': {'chrom': 'chr21',
                            'pos': 6789,
                            'coef': -0.91,
                            'major': 'C',
                            'minor': 'T',
                            'std_err': 0.09678,
                            'maf': 0.0223}
                   }

        beta_sub = BetaSubscriber(variant_to_expected)
        beta_sub.handle(result1)
        beta_sub.handle(result2)

        self.assertEqual(variant_to_expected[v1].e_risk,
                         variant_to_expected[v1].o_risk)
        self.assertEqual(variant_to_expected[v1].e_coef,
                         variant_to_expected[v1].o_coef)

        self.assertEqual(variant_to_expected[v2].e_risk,
                         variant_to_expected[v2].o_risk)
        self.assertEqual(variant_to_expected[v2].e_coef,
                         variant_to_expected[v2].o_coef)

    def test_BetaSubscriber_handle_AlleleFlip(self):
        v1 = geneparse.Variant(None, 2, 12345, ["A", "G"])
        # flipped alleles and reversed coefficient sign of noFlip function
        b1 = BetaTuple("G", -0.45)
        v2 = geneparse.Variant(None, 21, 6789, ["C", "T"])
        # flipped alleles and reversed coefficient sign of noFlip function
        b2 = BetaTuple("C", 0.91)

        variant_to_expected = {v1: b1, v2: b2}

        result1 = {'MODEL': {'nobs': 3767},
                   'SNPs': {'chrom': 'chr2',
                            'pos': 12345,
                            'coef': 0.45,
                            'major': 'G',
                            'minor': 'A',
                            'std_err': 0.07027076759612147,
                            'maf': 0.060788425803026284}
                   }

        result2 = {'MODEL': {'nobs': 2021},
                   'SNPs': {'chrom': 'chr21',
                            'pos': 6789,
                            'coef': -0.91,
                            'major': 'C',
                            'minor': 'T',
                            'std_err': 0.09678,
                            'maf': 0.0223}
                   }

        beta_sub = BetaSubscriber(variant_to_expected)
        beta_sub.handle(result1)
        beta_sub.handle(result2)

        self.assertEqual(variant_to_expected[v1].e_risk,
                         variant_to_expected[v1].o_risk)
        self.assertEqual(variant_to_expected[v1].e_coef,
                         variant_to_expected[v1].o_coef)

        self.assertEqual(variant_to_expected[v2].e_risk,
                         variant_to_expected[v2].o_risk)
        self.assertEqual(variant_to_expected[v2].e_coef,
                         variant_to_expected[v2].o_coef)

    """This function tests de computation of beta coefficients by comparing
    beta coefficients computed with plink (--assoc --linear) as
    the expected coefficients and the program function (using genetest)
    as the observed coefficients"""
    def test_compute_beta_coefficient_linearPhenotype(self):
        # GET VARIANTS
        # Load variants from plink association results file
        assoc_df = pd.read_csv(
                resource_filename(__name__,
                                  "data/plink_association_results.linear"),
                delim_whitespace=True
            )

        # Load variants from plink bim file
        bim_df = pd.read_csv(
                resource_filename(__name__, "data/extract_tag_test.bim"),
                sep="\t",
                names=["chrom", "id", "posCM", "pos", "allele1", "allele2"]
            )
        # Remove variants with mising allele info from bim
        bim_df = bim_df[bim_df.allele1 != "0"]

        # Join dataframes
        df = pd.merge(
                bim_df,
                assoc_df,
                left_on=["chrom", "pos"],
                right_on=["CHR", "BP"],
                how="inner"
            )

        # Sort variants according to p-value
        df = df.sort_values("P")

        # Keep first 200 variants
        df = df.head(n=200)

        # Get the variants from rows
        plink_variants = {}
        for idx, row in df.iterrows():
            v = geneparse.Variant(row.id,
                                  row.chrom,
                                  row.pos,
                                  [row.allele1, row.allele2])
            plink_variants[v] = BetaTuple(row.A1, row.BETA)

        # GENOTYPES
        reader = geneparse.parsers["plink"](
                resource_filename(__name__, "data/extract_tag_test")
            )

        extractor = geneparse.Extractor(
                reader,
                variants=plink_variants.keys()
            )

        # COMPUTE OBSERVED COEFFICIENTS
        beta_sub = compute_beta_coefficients(
                resource_filename(__name__, "data/pheno.csv"),
                "pheno_val",
                "sample",
                " ",
                None,
                "linear",
                max(cpu_count()-1, 1),
                plink_variants,
                extractor
            )

        reader.close()

        # COMPARE EXPECTED AND OBSERVED COEFFICIENTS
        for v, b in beta_sub.variant_to_expected.items():
            self.assertEqual(b.e_risk, b.o_risk)
            self.assertAlmostEqual(b.e_coef, b.o_coef, places=3)
