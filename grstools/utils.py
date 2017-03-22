"""
Utilities to manage files.
"""

import logging

import pandas as pd
import numpy as np

from genetest.subscribers import ResultsMemory
from genetest.analysis import execute_formula
from genetest.phenotypes.text import TextPhenotypes


logger = logging.getLogger(__name__)


COL_TYPES = {
    "name": str, "chrom": str, "pos": int, "reference": str, "risk": str,
    "p-value": float, "effect": float
}


def parse_computed_grs_file(filename):
    return pd.read_csv(filename, sep=",", index_col="sample")


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

    Returns:
        A pandas dataframe.

    """
    df = pd.read_csv(filename, sep=sep, dtype=COL_TYPES)

    cols = list(COL_TYPES.keys())

    # Optional columns.
    if "maf" in df.columns:
        cols.append("maf")

    # This will raise a KeyError if needed.
    df = df[cols]

    # Make the alleles uppercase.
    df["reference"] = df["reference"].str.upper()
    df["risk"] = df["risk"].str.upper()

    # Apply thresholds.
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


def mr_effect_estimate(phenotypes, outcome, exposure):
    """Estimate the effect of the exposure on the outcome using the ratio
    method.
    """
    logger.warning("For now, continuous outcomes and exposures are assumed.")

    def _estimate_beta(phen):
        # Regress big_gamma = Y ~ G
        stats = regress("{} ~ grs".format(outcome), "linear", phen)
        big_gamma = stats["beta"]

        # Regress small_gamma = X ~ G
        stats = regress("{} ~ grs".format(exposure), "linear", phen)
        small_gamma = stats["beta"]

        # Ratio estimate is beta = big_gamma / small_gamma
        return big_gamma / small_gamma

    # Bootstrap standard error estimates.
    df = phenotypes._phenotypes

    beta = _estimate_beta(phenotypes)

    betas = []
    n = phenotypes.get_nb_samples()
    for i in range(1000):
        idx = np.random.choice(n, size=n, replace=True)
        phenotypes._phenotypes = df.iloc[idx, :]
        betas.append(_estimate_beta(phenotypes))

    # FIXME Should I return the mean beta estimate from the bootstrap?
    return beta, np.std(betas)


def _create_genetest_phenotypes(grs_filename, phenotypes_filename,
                                phenotypes_sample_column="sample",
                                phenotypes_separator=","):
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
