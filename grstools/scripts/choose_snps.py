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

import pandas as pd
import numpy as np

from .match_snps import ld


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


def main():
    # Parameters
    p_threshold = 5e-8
    maf_threshold = 0.05
    ld_threshold = 0.05

    out = []

    # Variant to significance (initialized as a list but will be an
    # OrderedDict).
    summary = []

    # Variant index for range queries.
    # chromosome to sorted list of positions.
    index = collections.defaultdict(list)

    # Variant to genotypes.
    genotypes = {}

    # Read the summary statistics.
    filename = "/Users/legaultmarc/projects/StatGen/hdl_grs/data/summary.txt"
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

    # Sort the variants in the index to allow bisecting.
    # We also remember the index positions which is useful for bisecting.
    index_positions = {}
    for chrom in index:
        index[chrom] = sorted(index[chrom], key=lambda x: x.pos)
        index_positions[chrom] = [i.pos for i in index[chrom]]

    # Convert the summary statistics to an ordereddict of loci to p-values.
    summary = collections.OrderedDict(sorted(summary, key=lambda x: x[1]))

    # Extract the genotypes for all the variants in the summary.
    reference = format_map["plink"](
        "/Users/legaultmarc/projects/StatGen/grs/test_data/big",
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

    while len(summary) > 0:
        cur, p = summary.popitem(last=False)
        if cur not in genotypes:
            continue

        out.append(cur)

        cur_geno = genotypes[cur]

        left, right = region_query(index_positions, cur, int(100e3))
        loci = index[cur.chrom][left:right]

        # Build the matrix.
        other_genotypes = []
        retained_loci = []

        # For all the loci in this region we match with the reference to
        # build a genotype matrix.
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

        # Only variant in locus.
        if other_genotypes.shape[0] == 0:
            print("Singleton: {}".format(cur))
            continue

        # Compute the LD in block.
        cur_nan = np.isnan(cur_geno)
        nans = np.isnan(other_genotypes)

        n = np.sum(~cur_nan) - nans.sum(axis=0)

        other_genotypes[nans] = 0
        cur_geno[cur_nan] = 0

        r2 = (np.dot(cur_geno, other_genotypes) / n) ** 2

        # Remove all the correlated variants.
        for variant, pair_ld in zip(retained_loci, r2):
            if pair_ld > ld_threshold:
                del summary[variant]

        print("Remaining {} variants.".format(len(summary)))

    with open("dump.pkl", "wb") as f:
        pickle.dump(out, f)
