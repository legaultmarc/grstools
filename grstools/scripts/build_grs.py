"""
Compute the GRS from genotypes, a GRS file and a name mapping.
"""

import os
import collections
import logging
import argparse
import pandas as pd

from genetest.genotypes import format_map
from genetest.genotypes.core import Representation


logger = logging.getLogger(__name__)

Alleles = collections.namedtuple("Alleles", ["minor", "major"])


class DiscordantAlleles(ValueError):
    pass


def compute_grs(grs, geno, alleles):
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
            raise DiscordantAlleles(
                "The alleles for the variant '{}' are different in the "
                "genotypes data and in the GRS file. Fix this manually to "
                "avoir the wrong allele from being used in the GRS "
                "computation."
            )

    # Compute the GRS.
    for variant in geno.columns:
        geno[variant] *= grs.loc[variant, "beta"]

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

    # Read mapper into a dict.
    mappings = {}
    with open(args.mapper, "r") as f:
        header = f.readline().strip().split(args.delimiter)
        header = {col: i for i, col in enumerate(header)}

        for line in f:
            line = line.strip().split(args.delimiter)
            mappings[line[header["source_name"]]] = line[header["target_name"]]

    variants = list(mappings.keys())

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

        alleles[variant] = Alleles(minor=g.minor, major=g.major)

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

            n = grs.shape[0]

            grs["target"] = grs.index.map(lambda x: mappings.get(x, x))
            grs = pd.merge(
                grs, info, left_on="target", right_index=True, how="left"
            )
            import numpy as np
            print(grs.loc[np.isnan(grs["genotype_confidence_weight"]), :])
            grs = grs.drop("target", axis=1)

            assert grs.shape[0] == n, "Some variants have no INFO scores."

        else:
            logger.warning(
                "Could not find INFO file to weight genotypes (looked at "
                "'{}')".format(info_filename)
            )

    logger.info("Writing file containing the GRS: '{}'".format(args.out))
    computed_grs = compute_grs(grs, geno, alleles)
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
        "--out",
        help="Filename for the computed GRS (default: %(default)s).",
        default="computed_grs.csv"
    )

    return parser.parse_args()
