"""
Test utilities module.
"""


import os
import unittest

import numpy as np
import pandas as pd
import geneparse

from ..utils import compute_ld


def get_ld_test_directory():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "data", "ld")
    )


def get_ld_test_plink_prefix(dataset_with_na=False):
    suffix = ".missing" if dataset_with_na else ""
    return os.path.join(
        get_ld_test_directory(), "common_extracted_1kg" + suffix
    )


def std_genotypes(g):
    """Standardize an additive genotypes vector."""
    return (g - np.nanmean(g)) / np.nanstd(g)


def bin_std_genotypes(g):
    """Standardize an additive genotype vector using expected value and
        variance from a binomial distribution.

    """
    freq = np.nanmean(g) / 2
    variances = 2 * freq * (1 - freq)

    return (g - 2 * freq) / np.sqrt(variances)


def get_rs1800775_std_genotypes(dataset_with_na=False):
    # Read reference genotypes
    reader = geneparse.parsers["plink"]
    with reader(get_ld_test_plink_prefix(dataset_with_na)) as reader:
        rs1800775 = reader.get_variant_by_name("rs1800775")

        # Make sure there is only one variant with that name in the file.
        assert len(rs1800775) == 1
        return std_genotypes(rs1800775[0].genotypes)


def get_other_std_genotypes(dataset_with_na=False):
    reader = geneparse.parsers["plink"]
    with reader(get_ld_test_plink_prefix(dataset_with_na)) as reader:
        other_names = []
        other_variants = []
        for g in reader.iter_genotypes():
            # We only keep other variants.
            if g.variant.name == "rs1800775":
                continue

            else:
                other_names.append(g.variant.name)
                other_variants.append(
                    std_genotypes(g.genotypes)
                )

        other_variants = np.array(other_variants).T
        assert other_variants.shape == (503, 491)

    return other_names, other_variants


def read_expected_ld(dataset_with_na=False):
    suffix = ".missing.ld" if dataset_with_na else ".ld"

    expected = pd.read_csv(
        os.path.join(
            get_ld_test_directory(), "plink_rs1800775_pairs" + suffix
        ),
        delim_whitespace=True
    )
    expected = expected.set_index("SNP_B", verify_integrity=True)
    return expected[["R2"]]


class TestUtilities(unittest.TestCase):
    def test_ld_computation(self):
        # The tested function computes LD between a SNP and all others.
        # rs1800775 is that SNP.
        rs1800775 = get_rs1800775_std_genotypes()

        # Extract all the other variants.
        other_names, other_variants = get_other_std_genotypes()

        # Compute the LD.
        ld_vector = compute_ld(rs1800775, other_variants, r2=True)

        # Make it a DF.
        assert len(other_names) == ld_vector.shape[0]
        observed = pd.DataFrame(
            ld_vector, index=other_names, columns=["observed_r2"]
        )

        # Read the expected LD.
        expected = read_expected_ld()

        # Join and compare.
        df = observed.join(expected, how="outer")
        df = df[["observed_r2", "R2"]]

        squared_error = np.mean(
            (df.observed_r2 - df.R2) ** 2
        )

        self.assertAlmostEqual(squared_error, 0)

    def test_ld_computation_with_na_values(self):
        rs1800775 = get_rs1800775_std_genotypes(dataset_with_na=True)

        other_names, other_variants = get_other_std_genotypes(
            dataset_with_na=True
        )

        ld_vector = compute_ld(rs1800775, other_variants, r2=True)

        assert len(other_names) == ld_vector.shape[0]
        observed = pd.DataFrame(
            ld_vector, index=other_names, columns=["observed_r2"]
        )

        expected = read_expected_ld(dataset_with_na=True)

        # Join and compare.
        df = observed.join(expected, how="outer")
        df = df[["observed_r2", "R2"]]

        squared_error = np.mean(
            (df.observed_r2 - df.R2) ** 2
        )

        self.assertAlmostEqual(squared_error, 0, places=5)
