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

import argparse
import logging
import bisect
import collections

from genetest.genotypes import format_map
from genetest.genotypes.core import Representation, MarkerInfo
from genetest.statistics.descriptive import get_maf

import numpy as np


logger = logging.getLogger(__name__)


class Variant(object):
    __slots__ = ("chrom", "pos", "alleles")

    def __init__(self, chrom, pos, alleles=None):
        self.chrom = str(chrom)
        self.pos = int(pos)

        if alleles is None or "." in alleles:
            self.alleles = "ANY"
        else:
            self.alleles = frozenset([i.lower() for i in alleles])

    def locus_eq(self, other):
        """Chromosome and position match."""
        return (
            self.chrom == other.chrom and
            self.pos == other.pos
        )

    def __eq__(self, other):
        """Equality is defined as being the same locus and matching alleles.

        Alleles are considered to match if they are the same letter or if
        the "." is used to denote any variant's allele (ANY).
        """
        if isinstance(other, MarkerInfo):
            return self.markerinfo_eq(other)

        return (self.locus_eq(other) and
                (self.alleles == other.alleles or
                 self.alleles == "ANY" or
                 other.alleles == "ANY"))

    def __hash__(self):
        return hash((self.chrom, self.pos, self.alleles))

    def __repr__(self):
        return "<Variant chr{}:{}_[{}]>".format(self.chrom, self.pos,
                                                ",".join(self.alleles))


def region_query(index, variant, padding):
    index = index[variant.chrom]
    left = bisect.bisect(index, variant.pos - padding // 2)
    right = bisect.bisect(index, variant.pos + padding // 2)
    return left, right


def read_summary_statistics(filename, p_threshold, sep="\t"):
    # Variant to significance (initialized as a list but will be an
    # OrderedDict).
    summary = []

    # Variant index for range queries.
    # chromosome to sorted list of positions.
    index = collections.defaultdict(list)

    with open(filename) as f:
        header = f.readline().strip().split(sep)
        header = {v: k for k, v in enumerate(header)}

        for line in f:
            line = line.strip().split(sep)

            # Locus information
            name = line[header["name"]]
            chrom = line[header["chrom"]]
            pos = int(line[header["pos"]])

            # Statistics
            p = float(line[header["p-value"]])
            effect = float(line[header["effect"]])

            # Alleles
            reference = line[header["reference"]]
            risk = line[header["risk"]]

            # Completely ignore variants that do not pass the threshold.
            if p > p_threshold:
                continue

            variant = Variant(chrom, pos, [reference, risk])

            # Add variant to the summary statistics.
            summary.append((variant, (p, name, effect, reference, risk)))

            # Keep an index for faster range queries.
            index[chrom].append(variant)

    # Sort the index.
    for chrom in index:
        index[chrom] = sorted(index[chrom], key=lambda x: x.pos)

    # Convert the summary statistics to an ordereddict of loci to stats.
    summary = collections.OrderedDict(sorted(summary, key=lambda x: x[1][0]))

    return summary, index


def extract_genotypes(filename, summary, maf_threshold):
    genotypes = {}

    # Extract the genotypes for all the variants in the summary.
    reference = format_map["plink"](
        filename,
        representation=Representation.ADDITIVE
    )

    for info in reference.iter_marker_info():
        reference_variant = Variant(info.chrom, info.pos, [info.a1, info.a2])

        if reference_variant in summary:
            g = reference.get_genotypes(info.marker).genotypes["geno"]
            maf, minor, major, flip = get_maf(g, info.a1, info.a2)

            if maf > maf_threshold:
                # Standardize.
                g = g.values
                g = (g - np.nanmean(g)) / np.nanstd(g)
                genotypes[reference_variant] = g

    return genotypes


def build_genotype_matrix(cur, loci, genotypes, summary):
    # Build the genotype matrix from all the neighbouring variants with
    # available genotypes. The retained loci list will contain the
    # corresponding varinat objects.
    other_genotypes = []
    retained_loci = []

    for locus in loci:
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


def greedy_pick_clump(summary, genotypes, index, ld_threshold, ld_window_size):
    out = []

    # Extract the positions from the index to comply with the bisect API.
    index_positions = {}
    for chrom in index:
        index_positions[chrom] = [i.pos for i in index[chrom]]

    while len(summary) > 0:

        # Get the next best variant.
        cur, info = summary.popitem(last=False)
        if cur not in genotypes:
            continue

        # Add it to the GRS.
        p, name, effect, reference, risk = info
        out.append((name, cur.chrom, cur.pos, reference, risk, effect))

        # Get the genotypes for the current variant.
        cur_geno = genotypes[cur]

        # Do a region query in the index to get neighbouring variants.
        left, right = region_query(index_positions, cur, ld_window_size)
        loci = index[cur.chrom][left:right]  # TODO Check the indexing.

        # Extract genotypes.
        other_genotypes, retained_loci = build_genotype_matrix(
            cur, loci, genotypes, summary
        )

        if len(retained_loci) == 0:
            logger.debug(
                "Variant has no neighbour with genotypes ({})".format(cur)
            )
            continue

        # Compute the LD between all the neighbouring variants and the current
        # variant.
        r2 = compute_ld(cur_geno, other_genotypes)

        # Remove all the correlated variants.
        for variant, pair_ld in zip(retained_loci, r2):
            if pair_ld > ld_threshold:
                del summary[variant]

        logger.debug("Remaining {} variants.".format(len(summary)))

    return out


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--p-threshold",
        help="P-value threshold for inclusion in the GRS (default: 5e-8).",
        default=5e-8
    )

    parser.add_argument(
        "--maf-threshold",
        help="Minimum MAF to allow inclusion in the GRS (default %(default)s).",
        default=0.05,
    )

    parser.add_argument(
        "--ld-threshold",
        help=("LD threshold for the clumping step. All variants in LD with "
              "variants in the GRS are excluded iteratively (default "
              "%(default)s)."),
        default=0.05,
    )

    parser.add_argument(
        "--ld-window-size",
        help=("Size of the LD window used to find correlated variants. "
              "Making this window smaller will make the execution faster but "
              "increases the chance of missing correlated variants "
              "(default 500kb)."),
        default=int(500e3),
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

    return parser.parse_args()


def main():
    args = parse_args()

    # Parameters
    p_threshold = args.p_threshold
    maf_threshold = args.maf_threshold
    ld_threshold = args.ld_threshold
    ld_window_size = args.ld_window_size

    summary_filename = args.summary
    reference_filename = args.reference
    output_filename = args.output

    # Read the summary statistics.
    summary, index = read_summary_statistics(summary_filename, p_threshold)

    genotypes = extract_genotypes(reference_filename, summary, maf_threshold)

    grs = greedy_pick_clump(summary, genotypes, index, ld_threshold,
                            ld_window_size)

    with open(output_filename, "w") as f:
        f.write("name,chrom,pos,reference,risk,effect\n")
        for tu in grs:
            f.write(",".join([str(i) for i in tu]))
            f.write("\n")
