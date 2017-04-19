"""
Compute the GRS from genotypes and a GRS file.
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


import collections
import logging
import argparse

import numpy as np
import pandas as pd
import geneparse

from ..utils import parse_grs_file


logger = logging.getLogger(__name__)

ScoreInfo = collections.namedtuple(
    "ScoreInfo", ["effect", "reference", "risk"]
)


def compute_grs(samples, genotypes_and_info, quality_weight=True,
                ignore_ambiguous=True):
    quality_weight_warned = False

    grs = None
    for g, info in genotypes_and_info:
        # Note: some people use the MAF instead of 0.
        g.genotypes[np.isnan(g.genotypes)] = 0

        if g.coded == info.risk and g.reference == info.reference:
            # No need to flip.
            pass

        elif g.coded == info.reference and g.reference == info.risk:
            g.flip()

        else:
            raise RuntimeError(
                "Invalid alleles should have been filtered out upstream."
            )

        # Warn if alleles are ambiguous.
        if ignore_ambiguous and g.variant.alleles_ambiguous():
            logger.warning(
                "AMBIGUOUS alleles for {} (ignoring)."
                "".format(g.variant)
            )
            continue

        cur = g.genotypes * info.effect

        # Weight by quality if available.
        if isinstance(g.variant, geneparse.ImputedVariant) and quality_weight:
            if not quality_weight_warned:
                quality_weight_warned = True
                logger.info("WEIGHTING score by genotype confidence.")

            cur *= g.variant.quality

        if grs is None:
            grs = cur
        else:
            grs += cur

    return pd.DataFrame(grs, index=samples, columns=["grs"])


def main():
    args = parse_args()

    # Parse the GRS.
    grs = parse_grs_file(args.grs)

    # Extract genotypes.
    if args.genotypes_format not in geneparse.parsers.keys():
        raise ValueError(
            "Unknown reference format '{}'. Must be a genetest compatible "
            "format ({})."
            "".format(args.genotypes_format, list(geneparse.parsers.keys()))
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

    reader = geneparse.parsers[args.genotypes_format]
    reader = reader(
        args.genotypes,
        **genotypes_kwargs
    )

    genotypes_and_info = []
    for name, row in grs.iterrows():
        v = geneparse.Variant(
            name, row.chrom, row.pos, [row.reference, row.risk]
        )

        info = ScoreInfo(
            row.effect, row.reference, row.risk
        )

        # Get the genotype.
        g = reader.get_variant_genotypes(v)

        if len(g) == 0:
            logger.warning(
                "Excluding {} (no available genotypes)."
                "".format(v)
            )
        elif len(g) == 1:
            genotypes_and_info.append((g[0], info))
        else:
            logger.warning(
                "Excluding {} (duplicate variant or ambiguous multiallelic)"
                "".format(v)
            )

    computed_grs = compute_grs(
        reader.get_samples(),
        genotypes_and_info,
        quality_weight=not args.ignore_genotype_quality,
        ignore_ambiguous=not args.keep_ambiguous,
    )

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
        "--out",
        help="Filename for the computed GRS (default: %(default)s).",
        default="computed_grs.csv"
    )

    parser.add_argument(
        "--ignore-genotype-quality",
        help=("For imputed variants, if this flag is set, the variants "
              "will not be weighted with respect to their quality score. "
              "By default, the weights are used if available."),
        action="store_true"
    )

    parser.add_argument(
        "--keep-ambiguous",
        help=("Do not ignore ambiguous allele combinations (i.e. A/T and "
              "G/C). By default, such alleles are ignored."),
        action="store_true"
    )

    return parser.parse_args()
