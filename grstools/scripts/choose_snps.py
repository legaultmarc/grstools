"""
Choose SNPs from GWAS summary statistics.

Method 1:
    Sort by significance.
    Choose top SNP.
    Remove all SNPs in LD.
    Loop until p-value threshold or n variants is reached.

Ideas:
    - Skew GRS towards only biomarker increasing or decreasing alleles.

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


import argparse
import logging
import bisect
import collections

import geneparse

import numpy as np

from ..utils import parse_grs_file


debug = False

logger = logging.getLogger(__name__)
logger.debug("Starting")


class Row(object):
    __slots__ = ("name", "chrom", "pos", "reference", "risk", "p_value",
                 "effect", "maf")

    def __init__(self, name, chrom, pos, reference, risk, p_value, effect,
                 maf=None):
        """The row of a GRS file."""
        self.name = name
        self.chrom = chrom
        self.pos = pos
        self.reference = reference
        self.risk = risk
        self.p_value = p_value
        self.effect = effect
        self.maf = maf

    def write_header(self, f):
        f.write("name,chrom,pos,reference,risk,p-value,effect")
        if self.maf is not None:
            f.write(",maf")
        f.write("\n")

    @property
    def _fields(self):
        fields = [
            self.name, self.chrom, self.pos, self.reference, self.risk,
            self.p_value, self.effect
        ]

        if self.maf is not None:
            fields.append(self.maf)

        return fields

    def write(self, f, sep=","):
        for i, field in enumerate(self._fields):
            if i != 0:
                f.write(",")

            if type(field) is float:
                f.write("{:.9g}".format(field))
            else:
                f.write(str(field))

        f.write("\n")


def region_query(index, variant, padding):
    index = index[variant.chrom]
    left = bisect.bisect(index, variant.pos - padding // 2)
    right = bisect.bisect(index, variant.pos + padding // 2)
    return left, right


def _parse_region(s):
    message = "Expected format for region is: 'chr1:12345-22345'."
    if not s.startswith("chr"):
        raise ValueError(message)

    s = s[3:]
    try:
        chrom, tail = s.split(":")
        start, end = [int(i) for i in tail.split("-")]
    except:
        raise ValueError(message)

    # Flip start and end position if needed.
    if start > end:
        start, end = end, start

    return chrom, start, end


def read_summary_statistics(filename, p_threshold, sep=",",
                            keep_ambiguous=False, region=None,
                            exclude_region=None):
    if region is not None:
        region = _parse_region(region)
        logger.info("Only variants in region chr{}:{}-{} will be considered."
                    "".format(*region))

    if exclude_region is not None:
        exclude_region = _parse_region(exclude_region)
        logger.info("Only variants outside of region chr{}:{}-{} will be "
                    "considered.".format(*exclude_region))

    # Variant to stats orderedict (but constructed as a list).
    summary = []

    # Variant index for range queries.
    # chromosome to sorted list of positions.
    index = collections.defaultdict(list)

    df = parse_grs_file(filename, p_threshold=p_threshold, sep=sep)
    df.sort_values("p-value", inplace=True)

    # Method to see if a variant is in a region.
    def _in_region(variant, chrom, start, end):
        return (
            variant.chrom == chrom and
            start <= variant.pos <= end
        )

    # For now, this is not a limiting step, but it might be nice to parallelize
    # this eventually.
    for idx, info in df.iterrows():
        if info["p-value"] > p_threshold:
            break

        variant = geneparse.Variant(info["name"], info.chrom, info.pos,
                                    [info.reference, info.risk])

        # Region based inclusion/exclusion
        if region is not None:
            if not _in_region(variant, *region):
                continue

        if exclude_region is not None:
            if _in_region(variant, *exclude_region):
                continue
            

        if variant.alleles_ambiguous() and not keep_ambiguous:
            continue

        row_args = [info["name"], info.chrom, info.pos, info.reference,
                    info.risk, info["p-value"], info.effect]

        if "maf" in info.index:
            row_args.append(info.maf)

        row = Row(*row_args)
        summary.append((variant, row))
        index[info.chrom].append(variant)

    # Sort the index.
    for chrom in index:
        index[chrom] = sorted(index[chrom], key=lambda x: x.pos)

    # Convert the summary statistics to an ordereddict of loci to stats.
    summary = collections.OrderedDict(
        sorted(summary, key=lambda x: x[1].p_value)
    )

    return summary, index


def extract_genotypes(filename, summary, maf_threshold):
    genotypes = {}

    # Extract the genotypes for all the variants in the summary.
    reference = geneparse.parsers["plink"](filename)

    for variant, stats in summary.items():
        # Check if MAF is already known.
        if stats.maf is not None:
            if stats.maf < maf_threshold:
                continue

        ref_geno = reference.get_variant_genotypes(variant)

        if len(ref_geno) == 0:
            logger.warning("No genotypes for {}.".format(variant))

        elif len(ref_geno) == 1:
            g = ref_geno[0].genotypes

            # Compute the maf.
            mean = np.nanmean(g)
            maf = mean / 2

            if maf >= maf_threshold:
                # Standardize.
                g = (g - mean) / np.nanstd(g)
                genotypes[variant] = g

        else:
            logger.warning(
                "Ignoring {} (multiallelic or dup)."
                "".format(variant)
            )

    return genotypes


def build_genotype_matrix(cur, loci, genotypes, summary):
    """Build the genotype matrix of neighbouring variants.

    This will return a tuple containing the genotype matrix and a list of
    variant objects corresponding to the columns of the matrix.

    """
    other_genotypes = []
    retained_loci = []

    for locus in loci:
        # Get the genotypes in the reference.
        geno = genotypes.get(locus)

        if locus == cur:
            continue

        if geno is None:
            # Remove variants that have no genotype data in the reference
            # or that failed because of MAF thresholds.
            try:
                del summary[locus]
            except KeyError:
                pass

        # We only keep loci that have not been excluded yet.
        elif locus in summary:
            other_genotypes.append(geno)
            retained_loci.append(locus)

    other_genotypes = np.array(other_genotypes).T

    return other_genotypes, retained_loci


def compute_ld(cur_geno, other_genotypes):
    # Compute the LD in block.
    cur_nan = np.isnan(cur_geno)
    nans = np.isnan(other_genotypes)

    n = np.sum(~cur_nan) - nans.sum(axis=0)

    other_genotypes[nans] = 0
    cur_geno[cur_nan] = 0

    return (np.dot(cur_geno, other_genotypes) / n) ** 2


def greedy_pick_clump(summary, genotypes, index, ld_threshold, ld_window_size,
                      target_n=None):
    out = []

    # Extract the positions from the index to comply with the bisect API.
    index_positions = {}
    for chrom in index:
        index_positions[chrom] = [i.pos for i in index[chrom]]

    while len(summary) > 0:

        # Get the next best variant.
        cur, info = summary.popitem(last=False)
        logger.debug("CUR <- {} (p={})".format(cur, info.p_value))
        if cur not in genotypes:
            logger.debug("\tNEXT (no genotypes)")
            continue

        # Add it to the GRS.
        out.append(info)

        # Get the genotypes for the current variant.
        cur_geno = genotypes[cur]

        # Do a region query in the index to get neighbouring variants.
        left, right = region_query(index_positions, cur, ld_window_size)
        loci = index[cur.chrom][left:right]

        # Extract genotypes.
        other_genotypes, retained_loci = build_genotype_matrix(
            cur, loci, genotypes, summary
        )

        if len(retained_loci) == 0:
            logger.debug("\tNO_CLUMP (variant alone)")
            continue

        # Compute the LD between all the neighbouring variants and the current
        # variant.
        r2 = compute_ld(cur_geno, other_genotypes)

        # Remove all the correlated variants.
        logger.debug(
            "\tLD_REGION FROM {} TO {} ({}:{}-{})"
            "".format(retained_loci[0], retained_loci[-1], cur.chrom,
                      cur.pos - ld_window_size // 2,
                      cur.pos + ld_window_size // 2)
        )
        logger.debug("\tN_LD_CANDIDATES = {}".format(len(retained_loci)))

        for variant, pair_ld in zip(retained_loci, r2):
            if pair_ld > ld_threshold:
                if debug:
                    logger.debug(
                        "\tCLUMPING {} (R2={})".format(variant, pair_ld)
                    )
                del summary[variant]

        logger.debug("Chose {} variants (cur: {}), remaining {}.".format(
            len(out), info.name, len(summary)
        ))

        if target_n is not None and len(out) >= target_n:
            logger.debug("Target number of variants reached.")
            break

    return out


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--p-threshold",
        help="P-value threshold for inclusion in the GRS (default: 5e-8).",
        default=5e-8,
        type=float
    )

    parser.add_argument(
        "--target-n",
        help="Target number of variants to include in the GRS.",
        default=None,
        type=int
    )

    parser.add_argument(
        "--maf-threshold",
        help="Minimum MAF to allow inclusion in the GRS (default %(default)s).",
        default=0.05,
        type=float
    )

    parser.add_argument(
        "--ld-threshold",
        help=("LD threshold for the clumping step. All variants in LD with "
              "variants in the GRS are excluded iteratively (default "
              "%(default)s)."),
        default=0.05,
        type=float
    )

    parser.add_argument(
        "--ld-window-size",
        help=("Size of the LD window used to find correlated variants. "
              "Making this window smaller will make the execution faster but "
              "increases the chance of missing correlated variants "
              "(default 500kb)."),
        default=int(500e3),
        type=int
    )

    parser.add_argument(
        "--region",
        help=("Only consider variants located within a genomic region. "
              "The expected format is 'chrCHR:START-END'. For example: "
              "'chr1:12345-22345'."),
        default=None,
        type=str
    )

    parser.add_argument(
        "--exclude-region",
        help=("Only consider variants located OUTSIDE a genomic region. "
              "The expected format is 'chrCHR:START-END'. For example: "
              "'chr1:12345-22345'."),
        default=None,
        type=str
    )

    parser.add_argument(
        "--keep-ambiguous-alleles",
        help="Do not filter out ambiguous alleles (e.g. G/C or A/T)",
        action="store_true"
    )

    # Files
    parser.add_argument(
        "--summary",
        help=("Path to the summary statistics files. Required columns are "
              "'name', 'chrom', 'pos', 'p-value', 'effect', 'reference' and "
              "'risk'."),
        required=True
    )

    parser.add_argument(
        "--reference",
        help=("Path the binary plink file containing reference genotypes. "
              "These genotypes will be used for LD clumping."),
        required=True
    )

    parser.add_argument(
        "--output", "-o",
        help="Output filename (default: %(default)s).",
        default="selected.grs"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    return parser.parse_args()


def main():
    global debug

    args = parse_args()

    if args.debug:
        debug = True
        logger.setLevel(logging.DEBUG)

    # Parameters
    p_threshold = args.p_threshold
    target_n = args.target_n
    maf_threshold = args.maf_threshold
    ld_threshold = args.ld_threshold
    ld_window_size = args.ld_window_size
    keep_ambiguous = args.keep_ambiguous_alleles
    region = args.region
    exclude_region = args.exclude_region

    summary_filename = args.summary
    reference_filename = args.reference
    output_filename = args.output

    # Read the summary statistics.
    logger.info("Reading summary statistics.")
    summary, index = read_summary_statistics(summary_filename, p_threshold,
                                             keep_ambiguous=keep_ambiguous,
                                             region=region,
                                             exclude_region=exclude_region)

    logger.info("Extracting genotypes.")
    genotypes = extract_genotypes(reference_filename, summary, maf_threshold)

    logger.info(
        "Performing greedy selection with {} candidates."
        "".format(len(summary))
    )
    grs = greedy_pick_clump(summary, genotypes, index, ld_threshold,
                            ld_window_size, target_n)

    if len(grs) == 0:
        logger.warning(
            "No variant satisfied the provided thresholds (could not generate "
            "a GRS)."
        )
        return

    logger.info(
        "Writing the file containing the selected {} variants."
        "".format(len(grs))
    )

    with open(output_filename, "w") as f:
        grs[0].write_header(f)
        for row in grs:
            row.write(f)
