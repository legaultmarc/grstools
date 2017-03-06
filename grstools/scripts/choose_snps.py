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

import pickle
import bisect
import collections

from genetest.genotypes import format_map
from genetest.genotypes.core import Representation, MarkerInfo
from genetest.statistics.descriptive import get_maf

import numpy as np


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


def read_summary_statistics(filename, p_threshold):
    # Variant to significance (initialized as a list but will be an
    # OrderedDict).
    summary = []

    # Variant index for range queries.
    # chromosome to sorted list of positions.
    index = collections.defaultdict(list)

    with open(filename) as f:
        f.readline()  # Header
        for line in f:
            chrom, pos, rsid, a1, a2, beta, se, p = line.strip().split("\t")
            p = float(p)

            # Completely ignore variants that do not pass the threshold.
            if p > p_threshold:
                continue

            variant = Variant(chrom, pos, [a1, a2])

            # Add variant to the summary statistics.
            summary.append((variant, p))

            # Keep an index for faster range queries.
            index[chrom].append(variant)

    # Sort the index.
    for chrom in index:
        index[chrom] = sorted(index[chrom], key=lambda x: x.pos)

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


def greedy_pick_clump(summary, genotypes, index, ld_threshold):
    out = []

    # Extract the positions from the index to comply with the bisect API.
    index_positions = {}
    for chrom in index:
        index_positions[chrom] = [i.pos for i in index[chrom]]

    while len(summary) > 0:

        # Get the next best variant.
        cur, p = summary.popitem(last=False)
        if cur not in genotypes:
            continue

        # Add it to the GRS.
        out.append(cur)

        # Get the genotypes for the current variant.
        cur_geno = genotypes[cur]

        # Do a region query in the index to get neighbouring variants.
        left, right = region_query(index_positions, cur, int(100e3))
        loci = index[cur.chrom][left:right]  # TODO Check the indexing.

        # Extract genotypes.
        other_genotypes, retained_loci = build_genotype_matrix(
            cur, loci, genotypes, summary
        )

        if len(retained_loci) == 0:
            print("Variant has no neighbour with genotypes ({})".format(cur))
            continue

        # Compute the LD between all the neighbouring variants and the current
        # variant.
        r2 = compute_ld(cur_geno, other_genotypes)

        # Remove all the correlated variants.
        for variant, pair_ld in zip(retained_loci, r2):
            if pair_ld > ld_threshold:
                del summary[variant]

        print("Remaining {} variants.".format(len(summary)))


def main():
    # Parameters
    p_threshold = 5e-8
    maf_threshold = 0.05
    ld_threshold = 0.05

    # Read the summary statistics.
    filename = "/Users/legaultmarc/projects/StatGen/hdl_grs/data/summary.txt"
    summary, index = read_summary_statistics(filename, p_threshold)

    # Convert the summary statistics to an ordereddict of loci to p-values.
    summary = collections.OrderedDict(sorted(summary, key=lambda x: x[1]))

    genotypes = extract_genotypes(
        "/Users/legaultmarc/projects/StatGen/grs/test_data/big",
        summary,
        maf_threshold
    )

    grs = greedy_pick_clump(summary, genotypes, index, ld_threshold)

    with open("dump.pkl", "wb") as f:
        pickle.dump(grs, f)
