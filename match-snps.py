#!/usr/bin/env python

"""
Utility to match lists of SNPs and to find tags if needed.
"""


import os
import argparse
import collections

import pandas as pd
from gepyto.structures.region import Region
from genetest.genotypes import format_map
from genetest.genotypes.core import Representation


Locus = collections.namedtuple("Locus", ("chrom", "pos"))


class Results(object):
    def __init__(self, filename, header):
        # Open the output file.
        self.f = open(
            os.path.join(os.path.dirname(__file__), filename),
            "w"
        )
        self.f.write("\t".join(header) + "\n")

    def write(self, *args):
        self.f.write(
            "\t".join([str(i) for i in args]) + "\n"
        )

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _cast_cols(*args):
    """Cast the columns of the variants dataframe to have the expected types.

    """
    for i in range(len(args)):
        args[i]["pos"] = args[i]["pos"].astype(int)
        args[i]["chrom"] = args[i]["chrom"].astype(str)
        args[i]["name"] = args[i]["name"].astype(str)

    return args


def _extend_with_complement(alleles):
    """Extend a set of alleles by there complement."""
    table = str.maketrans(
        "atgc",
        "tacg"
    )
    complement = {str.translate(i.lower(), table) for i in alleles}
    return alleles | complement


def match_name(results, warn, row, target):
    """Match variant by name in the source and target lists."""
    hit = target.loc[target["name"] == row["name"], :]

    if hit.shape[0] == 1:
        results.write(row["name"], row["name"], "NAME_MATCH")
        return True
    elif hit.shape[0] > 0:
        warn.write(
            row["name"],
            "NAME_MATCH",
            "{} matches by name (ambiguous)".format(hit.shape[0])
        )
    return False


def match_variant(results, warn, row, target):
    """Match variant by chromosome position and alleles."""
    alleles = {i.lower() for i in (row.a1, row.a2)}
    alleles = _extend_with_complement(alleles)

    hit = target.loc[
        (target["chrom"] == row.chrom) &
        (target["pos"] == row.pos) &
        (target["a1"].str.lower().isin(alleles)) &
        (target["a2"].str.lower().isin(alleles))
    ]

    if hit.shape[0] == 1:
        results.write(row.name, hit.iloc[0, :]["name"], "VARIANT_MATCH")
        return True
    elif hit.shape[0] > 0:
        warn.write(
            row["name"],
            "VARIANT_MATCH",
            "{} matches by variant (ambiguous)".format(hit.shape[0])
        )
    return False


def find_tags(missing_idx, source, target, reference, reference_format):
    """Find tags in the reference dataset."""
    LD_WINDOW = 100e3  # 100kb

    regions = {}

    # Build one region per chromosome.
    for idx, row in source.loc[missing_idx, :].iterrows():
        region = Region(
            row["chrom"],
            min(0, row["pos"] - LD_WINDOW // 2),
            row["pos"] + LD_WINDOW // 2,
        )

        if regions.get(row["chrom"]) is None:
            regions[row["chrom"]] = region
        else:
            regions[row["chrom"]] = regions[row["chrom"]].union(region)

    # Extract genotypes in every region.
    if reference_format not in format_map.keys():
        raise ValueError(
            "Unknown reference format '{}'. Must be a genetest compatible "
            "format ({}).".format(reference_format, list(format_map.keys()))
        )

    container = format_map[reference_format](reference,
                                             Representation.ADDITIVE)

    genotypes = {}
    for snp in container.iter_marker_genotypes():
        chrom = snp["chrom"]
        if Locus(chrom, snp["pos"]) in regions[chrom]:
            # Add the genotypes.
            if genotypes.get(chrom) is None:
                genotypes[chrom] = snp.genotypes
                genotypes[chrom].columns = [snp.name]

            else:
                genotypes[chrom].loc[
                    snp.genotypes.index,
                    snp.name
                ] = snp.genotypes

    # Save all the genotypes to disk.
    for chrom, genotypes in genotypes.items():
        genotypes.to_csv(
            os.path.join("genotypes", "chrom{}.genotypes.csv")
        )


def main(args):
    # Read the source file.
    cols = ["name", "chrom", "pos", "a1", "a2"]

    source = pd.read_csv(args.source, names=cols, header=0)
    target = pd.read_csv(args.target, names=cols, header=0)

    source, target = _cast_cols(source, target)

    # n the number of variants in the source.
    n = source.shape[0]

    results_cols = ["source_name", "target_name", "method"]
    warn_cols = ["source_name", "method", "message"]

    matches = 0
    missing = []
    with Results("matcher.output.txt", results_cols) as out, \
         Results("matcher.warnings.txt", warn_cols) as warn:
        for idx, row in source.iterrows():
            matched = False
            for matcher in MATCHERS:
                if matcher(out, warn, row, target):
                    matches += 1
                    matched = True
                    break

            if not matched:
                # We need to try to match in the reference dataset.
                missing.append(idx)

    if args.reference:
        find_tags(missing, source, target, args.reference,
                  args.reference_format)

    print("Done matching variants. Found hits for {} / {}.".format(matches, n))


def parse_args():
    description = (
        "Tool to match genetic variants from a source list to a target list. "
        "A use case it to apply a genetic risk score on a dataset based on "
        "a different set of genetic variants. "
        "The input file format is currently inflexible. Both the source and "
        "the target input files should have the following columns: "
        "name,chrom,pos,a1,a2 "
        "A header is also expected."
    )

    parser = argparse.ArgumentParser(description=description)

    # Information on the source file.
    parser.add_argument(
        "--source",
        type=str,
        help=("Source list of variants. This is the list of variants to match "
              "or tag in the target list."),
        required=True
    )

    parser.add_argument(
        "--target",
        type=str,
        help="Target list of variants.",
        required=True
    )

    parser.add_argument(
        "--reference",
        type=str,
        help=("Reference file containing genotypes used to infer LD and to "
              "find tag SNPs."),
        default=None
    )

    parser.add_argument(
        "--reference-format",
        type=str,
        help=("File format of the genotypes in the reference (default: "
              "%(default)s)."),
        default="plink"
    )

    return parser.parse_args()


MATCHERS = [match_name, match_variant]


if __name__ == "__main__":
    main(parse_args())
