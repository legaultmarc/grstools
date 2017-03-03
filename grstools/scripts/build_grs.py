"""
Compute the GRS from genotypes, a GRS file and a name mapping.
"""

import os
import collections
import logging
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from genetest.genotypes import format_map
from genetest.genotypes.core import Representation

from .match_snps import ld


logger = logging.getLogger(__name__)

Alleles = collections.namedtuple("Alleles", ["minor", "major"])


class DiscordantAlleles(ValueError):
    pass


class IncompleteMapping(Exception):
    pass


def compute_grs(grs, geno, alleles, skip_bad_alleles=False):
    # Set missing genotypes to zero (note that some people set it to the MAF).
    geno[geno.isnull()] = 0

    for variant, a in alleles.items():
        reference, risk = grs.loc[variant, ["reference", "risk"]].values
        if a.minor == risk and a.major == reference:
            # No need to flip and alleles are correct.
            pass
        elif a.minor == reference and a.major == risk:
            # Need to flip.
            geno[variant] = 2.0 - geno[variant]
        else:
            # Alleles are discordant.
            if skip_bad_alleles:
                logger.warning(
                    "EXCLUDING variant '{}' because of allele mismatch "
                    "between the genotypes file {} and the GRS file {}."
                    "".format(variant, (a.major, a.minor), (reference, risk))
                )
                geno[variant] = 0
                continue

            raise DiscordantAlleles(
                "The alleles for the variant '{}' are different in the "
                "genotypes data {} and in the GRS file {}. Fix this "
                "manually to avoid the wrong allele from being used in the "
                "GRS computation."
                "".format(variant, (a.major, a.minor), (reference, risk))
            )

    # Compute the GRS.
    geno_confidence_weight = "genotype_confidence_weight" in grs.columns
    if geno_confidence_weight:
        logger.info("WEIGHTING score by genotype confidence (e.g. INFO score)")

    for variant in geno.columns:
        geno[variant] *= grs.loc[variant, "beta"]

        if geno_confidence_weight:
            geno[variant] *= grs.loc[variant, "genotype_confidence_weight"]

    computed_grs = geno.sum(axis=1)
    computed_grs.name = "grs"
    return computed_grs


def main():
    args = parse_args()

    grs = pd.read_csv(args.grs, sep=args.delimiter)
    try:
        grs = grs[["name", "beta", "reference", "risk"]]
    except KeyError:
        raise KeyError(
            "Expected columns 'name', 'beta', 'reference' and 'risk' in the "
            "GRS file."
        )

    grs = grs.set_index("name", verify_integrity=True)
    grs["reference"] = grs["reference"].str.lower()
    grs["risk"] = grs["risk"].str.lower()

    # Read mapper into a dict.
    # The resulting dict is from source -> target.
    mappings = {}
    with open(args.mapper, "r") as f:
        header = f.readline().strip().split(args.delimiter)
        header = {col: i for i, col in enumerate(header)}

        for line in f:
            line = line.strip().split(args.delimiter)
            mappings[line[header["source_name"]]] = line[header["target_name"]]

    variants = list(mappings.keys())

    # Make sure the mapping has all the variants from the GRS.
    # The variants from the GRS are expected to be in the SOURCE.
    missing = set(grs.index) - set(mappings.keys())
    if missing:
        raise IncompleteMapping(
            "The provided mapping is incomplete (missing: {}). "
            "Fix this by removing unknown variants from the GRS or by adding "
            "mappings for them in the mapping file."
            "".format(tuple(missing))
        )

    # Extract genotypes in every region.
    if args.genotypes_format not in format_map.keys():
        raise ValueError(
            "Unknown reference format '{}'. Must be a genetest compatible "
            "format ({})."
            "".format(args.genotypes_format, list(format_map.keys()))
        )

    genotypes_kwargs = {}
    if args.genotypes_kwargs:
        for argument in args.genotypes_kwargs.split(","):
            key, value = argument.strip().split("=")

            if value.startswith("int:"):
                value = int(value[4:])

            elif value.startswith("float:"):
                value = float(value[6:])

            genotypes_kwargs[key] = value

    container = format_map[args.genotypes_format]
    container = container(
        args.genotypes, representation=Representation.ADDITIVE,
        **genotypes_kwargs
    )

    alleles = {}
    geno = None
    for variant in variants:
        # Get the genotype.
        g = container.get_genotypes(mappings[variant])
        if geno is None:
            geno = g.genotypes
            geno.columns = [variant]
        else:
            geno[variant] = g.genotypes

        alleles[variant] = Alleles(minor=g.info.get_minor().lower(),
                                   major=g.info.get_major().lower())

    # Create the genotype_confidence_weight column.
    if args.genotypes_format == "impute2":
        info_filename = args.genotypes.split(".impute2")[0] + ".impute2_info"
        if os.path.isfile(info_filename):
            logger.info(
                "Setting genotypes weight with respect to the INFO score "
                "foud in file '{}'".format(info_filename)
            )
            info = pd.read_csv(info_filename, sep="\t")
            info = info.set_index("name", verify_integrity=True)
            info = info[["info"]]
            info.columns = ["genotype_confidence_weight"]

            grs["target"] = grs.index.map(lambda x: mappings.get(x, x))
            grs = pd.merge(
                grs, info, left_on="target", right_index=True, how="left"
            )
            no_info = grs.loc[
                grs["genotype_confidence_weight"].isnull(),
                "target"
            ]

            if no_info.shape[0]:
                logger.warning(
                    "Could not find the INFO score for variants {}. They will "
                    "not be weighted for imputation confidence (weight set "
                    "to 1)."
                    "".format(tuple(no_info))
                )
                grs.loc[no_info.index, "genotype_confidence_weight"] = 1

            grs = grs.drop("target", axis=1)

        else:
            logger.warning(
                "Could not find INFO file to weight genotypes (looked at "
                "'{}')".format(info_filename)
            )

    if args.ld_plot:
        _ld = ld(geno.values)
        plt.imshow(_ld)
        plt.colorbar()
        np.fill_diagonal(_ld, -np.inf)

        logger.info(
            "WRITING LD plot for the GRS (max off-diagnoal LD={:.2f})."
            "".format(np.max(_ld[~np.isnan(_ld)]))
        )

        plt.savefig("{}_ld.png".format(args.out), dpi=300)

    computed_grs = compute_grs(grs, geno, alleles, args.skip_bad_alleles)
    logger.info("WRITING file containing the GRS: '{}'".format(args.out))
    computed_grs.to_csv(args.out, header=True, index_label="sample")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute the risk score.")

    parser.add_argument(
        "--genotypes",
        help="File containing genotype data.",
        type=str
    )

    parser.add_argument(
        "--genotypes-format",
        type=str,
        help=("File format of the genotypes in the reference (default: "
              "%(default)s)."),
        default="plink"
    )

    parser.add_argument(
        "--genotypes-kwargs",
        type=str,
        help=("Keyword arguments to pass to the genotypes container. "
              "A string of the following format is expected: "
              "'key1=value1,key2=value2,..."
              "It is also possible to prefix the values by 'int:' or 'float:' "
              "to cast the them before passing them to the constructor.")
    )

    parser.add_argument(
        "--grs",
        help=("File describing the variants in the GRS. "
              "The expected columns are 'name', 'beta', 'reference', 'risk'."),
        type=str,
        required=True
    )

    parser.add_argument(
        "--mapper",
        help=("File mapping variant names from the genotypes file to the GRS "
              "file. Expected columns are 'source_name' and 'target_name'."),
        type=str
    )

    parser.add_argument(
        "--delimiter", "-d",
        help=("Column delimiter for the grs and the mapper files."),
        default=","
    )

    parser.add_argument(
        "--skip-bad-alleles",
        help=("Skip variants whose alleles are inconsistent between the GRS "
              "and the genotypes files."),
        action="store_true"
    )

    parser.add_argument(
        "--ld-plot",
        help=("Add an LD plot to see how correlated the variants in the GRS "
              "are."),
        action="store_true"
    )

    parser.add_argument(
        "--out",
        help="Filename for the computed GRS (default: %(default)s).",
        default="computed_grs.csv"
    )

    return parser.parse_args()
