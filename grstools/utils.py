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


import sqlite3
import logging

import pandas as pd
import numpy as np
import scipy.stats

import geneparse.utils
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
    test_kwargs = None if test == "logistic" else {
        "condition_value_t": float("infinity")
    }

    execute_formula(
        phenotypes, None, model, test,
        test_kwargs=test_kwargs,
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

    # Filter suitable tags i.e. unambiguous and common enough.
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
            "Could not find tags for variant: {} (not in reference panel)."
            "".format(variant)
        )
        return None

    # Compute the LD.
    r = geneparse.utils.compute_ld(genotypes[idx], genotypes, r2=False).values
    r[idx] = 0

    best_tag = np.argmax(r ** 2)

    return genotypes[idx], genotypes[best_tag], r[best_tag]


def parse_kwargs(s):
    if s is None:
        return None

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


class InMemoryGenotypeExtractor(object):
    """Like geneparse's Extractor class but that loads everything in memory.

    Note that this class does nothing to the geneparse reader after reading.
    It is the caller's responsibility manage (close) the reader.

    """
    def __init__(self, parser, variants):
        self.variants = variants
        self.rowid_to_genotypes = {}
        self.variant_to_genotypes = {}

        self._not_found = set()
        self._dups_or_multi = set()

        self.con = sqlite3.connect(":memory:")
        self.cur = self.con.cursor()

        self._load_variant_in_memory(parser)

    def close(self):
        self.con.close()

    def _load_variant_in_memory(self, parser):
        self.cur.execute(
            "CREATE TABLE variants ("
            "  chrom TEXT, "
            "  pos INT"
            ")"
        )

        self.cur.execute(
            "CREATE INDEX _locus_idx ON variants (chrom, pos)"
        )

        self.con.commit()

        MAX_BUFFER_SIZE = 2000
        buffer = []
        rowid = 1
        for v in self.variants:
            # Get the variant from the parser.
            g = parser.get_variant_genotypes(v)

            if len(g) == 0:
                self._not_found.add(v)

            elif len(g) > 1:
                self._dups_or_multi.add(v)

            # If found and unique, add it to the cache and the database.
            else:
                g = g[0]
                buffer.append((rowid, v.chrom.name, v.pos))
                self.rowid_to_genotypes[rowid] = g
                self.variant_to_genotypes[v] = g
                rowid += 1

            # Push to the database if buffer is full.
            if len(buffer) >= MAX_BUFFER_SIZE:
                self._do_inserts(buffer)
                buffer = []

        # Push the remaining data if needed.
        if len(buffer) > 0:
            self._do_inserts(buffer)

        self.con.commit()

    def _do_inserts(self, buffer):
        self.cur.execute("BEGIN TRANSACTION")
        for row in buffer:
            self.cur.execute(
                "INSERT INTO variants (rowid, chrom, pos) VALUES (?, ?, ?)",
                row
            )
        self.cur.execute("COMMIT")

    def get_variant_genotypes(self, v):
        if v in self.variant_to_genotypes:
            return [self.variant_to_genotypes[v], ]
        else:
            return []

    def get_variants_in_region(self, chrom, left, right):
        if isinstance(chrom, geneparse.Chromosome):
            chrom = chrom.name

        self.cur.execute(
            "SELECT rowid FROM variants "
            "WHERE "
            "  chrom = ? AND "
            "  pos >= ? AND "
            "  pos <= ?",
            (chrom, left, right)
        )

        for row in self.cur:
            rowid = row[0]
            yield self.rowid_to_genotypes[rowid]
