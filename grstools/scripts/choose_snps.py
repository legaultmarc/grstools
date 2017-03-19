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

import geneparse

import numpy as np

from ..utils import parse_grs_file


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug("Starting up")


def region_query(index, variant, padding):
    index = index[variant.chrom]
    left = bisect.bisect(index, variant.pos - padding // 2)
    right = bisect.bisect(index, variant.pos + padding // 2)
    return left, right


def read_summary_statistics(filename, p_threshold, sep=","):
    # Variant to stats orderedict (but constructed as a list).
    summary = []

    # Variant index for range queries.
    # chromosome to sorted list of positions.
    index = collections.defaultdict(list)

    df = parse_grs_file(filename, p_threshold=p_threshold, sep=sep)
    df.sort_values("p-value", inplace=True)

    # TODO If we're going to build the summary as a list, we should at least
    # make this parallel.
    for name, info in df.iterrows():
        if info["p-value"] > p_threshold:
            break

        variant = geneparse.Variant(name, info.chrom, info.pos,
                                    [info.reference, info.risk])

        summary.append((
            variant,
            (info["p-value"], name, info.effect, info.reference, info.risk)
        ))

        index[info.chrom].append(variant)

    # Sort the index.
    for chrom in index:
        index[chrom] = sorted(index[chrom], key=lambda x: x.pos)

    # Convert the summary statistics to an ordereddict of loci to stats.
    summary = collections.OrderedDict(sorted(summary, key=lambda x: x[1][0]))

    return summary, index


def extract_genotypes(filename, summary, maf_threshold):
    genotypes = {}

    # Extract the genotypes for all the variants in the summary.
    reference = geneparse.parsers["plink"](filename)

    for reference_variant in reference.iter_variants():
        if reference_variant in summary:
            g = reference.get_variant_genotypes(reference_variant)
            if len(g) == 0:
                raise ValueError("This should not happen.")
            elif len(g) == 1:
                g = g[0]
            else:
                # TODO
                print(reference_variant)
                for i in g:
                    print("->", i)
                logger.warning(
                    "Ignoring {}: Multiallelics or duplicated variants are "
                    "not handled.".format(reference_variant)
                )
                continue

            g = g.genotypes
            mean = np.nanmean(g)

            # TODO Division could be avoided by manipulating maf_threshold.
            maf = mean / 2

            if maf > maf_threshold:
                # Standardize.
                g = (g - mean) / np.nanstd(g)
                genotypes[reference_variant] = g

    return genotypes


def _worker_extract_genotypes(lock, queue, variant, reference, summary,
                              maf_threshold):
    if variant in summary:
        # Get the genotypes.
        with lock:
            g = reference.get_variant_genotypes
        if len(g) == 0:
            raise ValueError("This should not happen.")
        elif len(g) == 1:
            # Single variant match.
            g = g[0]
        else:
            # TODO Handle duplicates and multi-allelics.
            return

        g = g.genotypes
        mean = np.nanmean(g)

        maf = mean / 2

        if maf > maf_threshold:
            g = (g - mean) / np.nanstd(g)
            queue.put((variant, g))


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

    logger.debug(
        "Starting the greedy SNP selection with {} candidates."
        "".format(len(summary))
    )
    while len(summary) > 0:

        # Get the next best variant.
        cur, info = summary.popitem(last=False)
        if cur not in genotypes:
            logger.debug("No genotypes for {}.".format(cur))
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

        logger.debug("Chose {} variants, remaining {}.".format(
            len(out), len(summary)
        ))

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
    logger.info("Reading summary statistics.")
    summary, index = read_summary_statistics(summary_filename, p_threshold)

    logger.info("Extracting genotypes.")
    genotypes = extract_genotypes(reference_filename, summary, maf_threshold)

    logger.info("Perform greedy selection.")
    grs = greedy_pick_clump(summary, genotypes, index, ld_threshold,
                            ld_window_size)

    logger.info("Writing the file containing the final selection.")
    with open(output_filename, "w") as f:
        f.write("name,chrom,pos,reference,risk,effect\n")
        for tu in grs:
            f.write(",".join([str(i) for i in tu]))
            f.write("\n")
