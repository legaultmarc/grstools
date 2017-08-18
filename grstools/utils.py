"""
Utilities to manage files.
"""

# This file is part of grstools.
#
# The MIT License (MIT)
#
# Copyright (c) 2017 Marc-Andre Legault
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import logging

import pandas as pd
import numpy as np
import scipy.stats

from genetest.subscribers import ResultsMemory
from genetest.analysis import execute_formula
from genetest.phenotypes.text import TextPhenotypes


logger = logging.getLogger(__name__)


COL_TYPES = {
    "name": str, "chrom": str, "pos": int, "reference": str, "risk": str,
    "effect": float
}


def parse_computed_grs_file(filename):
    df = pd.read_csv(filename, sep=",", index_col="sample")
    df.index = df.index.astype(str)
    return df


def parse_grs_file(filename, p_threshold=1, maf_threshold=0, sep=",",
                   log=False):
    """Parse a GRS file.

    The mandatory columns are:
        - name (variant name)
        - chrom (chromosome, a str))
        - pos (position, a int)
        - reference (reference allele)
        - risk (effect/risk allele)
        - p-value (p-value, a float)
        - effect (beta or OR or other form of weight, a float)

    Optional columns are:
        - maf
        - p-value is optional if used to compute the GRS only.

    Returns:
        A pandas dataframe.

    """
    df = pd.read_csv(filename, sep=sep, dtype=COL_TYPES)

    cols = list(COL_TYPES.keys())

    # Optional columns.
    if "maf" in df.columns:
        cols.append("maf")

    if "p-value" in df.columns:
        cols.append("p-value")

    # This will raise a KeyError if needed.
    df = df[cols]

    # Make the alleles uppercase.
    df["reference"] = df["reference"].str.upper()
    df["risk"] = df["risk"].str.upper()

    # Apply thresholds.
    if "p-value" in df.columns:
        if log:
            logger.info("Applying p-value threshold (p <= {})."
                        "".format(p_threshold))

        df = df.loc[df["p-value"] <= p_threshold, :]

    if "maf" in df.columns:
        if log:
            logger.info("Applying MAF threshold (MAF >= {})."
                        "".format(maf_threshold))
        df = df.loc[df["maf"] >= maf_threshold, :]

    return df


def mr_effect_estimate(phenotypes, outcome, exposure, n_iter=1000,
                       y_g_test="linear", x_g_test="linear"):
    """Estimate the effect of the exposure on the outcome using the ratio
    method.
    """
    def _estimate_beta(phen):
        # Regress big_gamma = Y ~ G
        stats = regress("{} ~ grs".format(outcome), y_g_test, phen)
        big_gamma = stats["beta"]

        # Regress small_gamma = X ~ G
        stats = regress("{} ~ grs".format(exposure), x_g_test, phen)
        small_gamma = stats["beta"]

        # Ratio estimate is beta = big_gamma / small_gamma
        return big_gamma / small_gamma

    # Using the percentile method to compute a confidence interval.
    df = phenotypes._phenotypes
    beta = _estimate_beta(phenotypes)

    betas = np.empty(n_iter, dtype=float)
    n = phenotypes.get_nb_samples()
    for i in range(n_iter):
        idx = np.random.choice(n, size=n, replace=True)
        phenotypes._phenotypes = df.iloc[idx, :]
        betas[i] = _estimate_beta(phenotypes)

    # Find the critical values
    # 95% CI -> 2.5% and 97.5%
    low, high = np.percentile(betas, [2.5, 97.5])

    # p-value
    # This method to calculate the p-value is derived from:
    # An Introduction to the Bootstrap. 1993. doi:10.1007/978-1-4899-4541-9
    # Efron B., Tibshirani RJ.
    #
    # Section 15.4: Relationship of hypothesis tests to confidence intervals
    # and the bootstrap.
    # TODO verify...
    # p = np.sum(betas < 0) / n_iter

    return beta, low, high, None


# TODO remove this method.
def _create_genetest_phenotypes(grs_filename, phenotypes_filename,
                                phenotypes_sample_column="sample",
                                phenotypes_separator=","):
    logger.warning("_create_genetest_phenotypes is deprecated.")

    # Read the GRS.
    grs = TextPhenotypes(grs_filename, "sample", ",", "", False)

    # Read the other phenotypes.
    phenotypes = TextPhenotypes(
        phenotypes_filename,
        phenotypes_sample_column,
        phenotypes_separator, "", False
    )

    phenotypes.merge(grs)
    return phenotypes


def regress(model, test, phenotypes):
    """Regress a GRS on a phenotype."""
    subscriber = ResultsMemory()

    # Check that the GRS was included in the formula.
    if "grs" not in model:
        raise ValueError(
            "The grs should be included in the regression model. For example, "
            "'phenotype ~ grs + age' would be a valid model, given that "
            "'phenotype' and 'age' are defined in the phenotypes file."
        )

    # Make sure the test is linear or logistic.
    if test not in {"linear", "logistic"}:
        raise ValueError("Statistical test should be logistic or linear.")

    # Execute the test.
    execute_formula(
        phenotypes, None, model, test,
        test_kwargs=None,
        subscribers=[subscriber],
        variant_predicates=None,
    )

    # Get the R2, the beta, the CI and the p-value.
    results = subscriber.results
    if len(results) != 1:
        raise NotImplementedError(
            "Only simple, single-group regression models are supported."
        )
    results = results[0]

    out = {}

    out["beta"] = results["grs"]["coef"]
    out["CI"] = (results["grs"]["lower_ci"], results["grs"]["upper_ci"])
    out["p-value"] = results["grs"]["p_value"]

    if test == "linear":
        out["intercept"] = results["intercept"]["coef"]
        out["R2"] = results["MODEL"]["r_squared_adj"]

    return out


def find_tag(reader, variant, extract_reader=None, window_size=100e3,
             maf_threshold=0.01, sample_normalization=True):
    """Find tags for the variant in the given reference genotypes.

    extract_reader can be a second geneparse reader to make sure that the
    selected tags are also available in a given genotype dataset.

    """

    genotypes = reader.get_variants_in_region(
        variant.chrom,
        variant.pos - (window_size // 2),
        variant.pos + (window_size // 2)
    )

    # Take the subset of genotypes that are also available in the other
    # genetic dataset if provided.
    if extract_reader is not None:
        genotypes = [
            i for i in genotypes
            if len(extract_reader.get_variant_genotypes(i.variant)) == 1
        ]

    def _valid(g):
        return (
            (not g.variant.alleles_ambiguous() and g.maf() >= maf_threshold) or
            g.variant == variant
        )

    genotypes = [g for g in genotypes if _valid(g)]

    # There are no other variants in the region to be used as tags.
    if len(genotypes) < 2:
        return None

    # Find the index variant.
    idx = 0
    while idx < len(genotypes) - 1:
        if genotypes[idx].variant == variant:
            break
        else:
            idx += 1

    if genotypes[idx].variant != variant:
        logger.warning(
            "Could not find tags for missing variant: {}.".format(variant)
        )
        return None

    # Create the Matrix of normalized genotypes.
    # mat is a matrix of n_samples x n_snps
    mat = np.vstack(tuple((i.genotypes for i in genotypes))).T

    if sample_normalization:
        # Sample normalization
        mat = mat - np.nanmean(mat, axis=0)[np.newaxis, :]
        mat = mat / np.nanstd(mat, axis=0)[np.newaxis, :]

    else:
        # Normalization based on binomial distribution.
        freq = np.nanmean(mat, axis=0) / 2
        variances = np.sqrt(2 * freq * (1 - freq))

        mat = (mat - 2 * freq[np.newaxis, :]) / variances[np.newaxis, :]

    # Compute the LD.
    r = compute_ld(mat[:, idx], mat)
    r[idx] = 0

    best_tag = np.argmax(r ** 2)

    return genotypes[idx], genotypes[best_tag], r[best_tag]


def compute_ld(cur_geno, other_genotypes, r2=False):
    # Compute the LD in block.
    cur_nan = np.isnan(cur_geno)
    nans = np.isnan(other_genotypes)

    n = np.sum(~cur_nan) - nans.sum(axis=0)

    other_genotypes[nans] = 0
    cur_geno[cur_nan] = 0

    r = np.dot(cur_geno, other_genotypes) / n

    if r2:
        return r ** 2
    else:
        return r


def parse_kwargs(s):
    kwargs = {}
    for argument in s.split(","):
        key, value = argument.strip().split("=")

        if value.startswith("int:"):
            value = int(value[4:])

        elif value.startswith("float:"):
            value = float(value[6:])

        kwargs[key] = value

    return kwargs


def clopper_pearson_interval(k, n, alpha=0.001):
    """
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected
    successes on n trials.

    Clopper Pearson intervals are a conservative estimate.

    Implementation was adapted from https://gist.github.com/DavidWalz/8538435
    but it does exactly what is expected given the Wikipedia article.

    """
    lo = scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    hi = scipy.stats.beta.ppf(1 - alpha/2, k+1, n-k)

    return lo, hi
