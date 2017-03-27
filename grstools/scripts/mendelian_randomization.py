"""
Compute the GRS from genotypes and a GRS file.
"""

import logging
import argparse

from .evaluate import _add_phenotype_arguments
from ..utils import mr_effect_estimate, _create_genetest_phenotypes


logger = logging.getLogger(__name__)


def main():
    args = parse_args()

    phenotypes = _create_genetest_phenotypes(
        args.grs_filename, args.phenotypes_filename,
        args.phenotypes_sample_column, args.phenotypes_separator
    )

    if args.outcome_type == "continuous":
        y_g_test = "linear"
    elif args.outcome_type == "discrete":
        y_g_test = "logistic"
    else:
        raise ValueError(
            "Expected outcome type to be 'discrete' or 'continuous'."
        )

    if args.exposure_type == "continuous":
        x_g_test = "linear"
    elif args.exposure_type == "discrete":
        x_g_test = "logistic"
    else:
        raise ValueError(
            "Expected exposure type to be 'discrete' or 'continuous'."
        )

    n_iter = 1000
    logger.info(
        "Computing MR estimates using the ratio method. Bootstrapping "
        "standard errors can take some time."
    )
    beta, low, high = mr_effect_estimate(
        phenotypes, args.outcome, args.exposure, n_iter, y_g_test, x_g_test
    )

    print("The estimated beta of the exposure on the outcome and its 95% CI "
          "(computed using the empirical " "bootstrap) are:\n")
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
    parser.add_argument(
        "--exposure-type", type=str,
        help="Either continuous or discrete.",
        default="continuous"
    )
    parser.add_argument(
        "--outcome-type", type=str,
        help="Either continuous or discrete.",
        default="continuous"
    )
    _add_phenotype_arguments(parser)

    return parser.parse_args()
