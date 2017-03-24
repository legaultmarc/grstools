"""
Compute the GRS from genotypes and a GRS file.
"""

import logging
import argparse

import numpy as np
import pandas as pd
import geneparse

from .evaluate import _add_phenotype_arguments
from ..utils import mr_effect_estimate, _create_genetest_phenotypes


logger = logging.getLogger(__name__)


def main():
    args = parse_args()

    phenotypes = _create_genetest_phenotypes(
        args.grs_filename, args.phenotypes_filename,
        args.phenotypes_sample_column, args.phenotypes_separator
    )

    n_iter = 1000
    beta, low, high = mr_effect_estimate(
        phenotypes, args.outcome, args.exposure, n_iter=n_iter
    )

    print("We ran {} bootstrap iterations to estimate the Beta coefficient."
          "".format(n_iter))
    print("The estimated value and 95% CI (computed using the empirical "
          "bootstrap) are:\n")
    print("{:.4g} ({:.4g}, {:.4g})".format(beta, low, high))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the effect of an exposure on an outcome using "
            "a GRS with an effect on the exposure.\n"
            "Estimates are done using the ratio method."
        )
    )

    parser.add_argument("--grs-filename", type=str)
    parser.add_argument("--exposure", type=str)
    parser.add_argument("--outcome", type=str)
    _add_phenotype_arguments(parser)

    return parser.parse_args()
