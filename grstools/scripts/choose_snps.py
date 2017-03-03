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

from genetest.genotypes import format_map
from genetest.genotypes.core import Representation

import pandas as pd
import numpy as np

from .match_snps import ld


def parse_summary_statistics(filename, columns, sep=","):
    columns = columns.items()
    df = pd.read_csv(filename, sep=sep)

    df = df[[i[1] for i in columns]]
    df.columns = [i[0] for i in columns]

    return df.sort_values("significance")


def apply_significance_threshold(df, p):
    return df.loc[df["significance"] <= p, :]


def get_reference(filename, **kwargs):
    """Reads the reference panel in plink BED format."""
    return format_map["plink"](
        filename, representation=Representation.ADDITIVE, **kwargs
    )


def get_ld_candidates(df, chrom, pos, ld_window=None):
    if ld_window is None:
        ld_window = int(100e3)

    df = df.loc[
        (df["chrom"] == chrom) &
        (df["pos"] >= pos - (ld_window // 2)) &
        (df["pos"] <= pos + (ld_window // 2))
    ]

    # Find the position of the index SNP.
    index = df.loc[df["pos"] == pos].index
    if len(index) > 1:
        raise ValueError("SNP is ambiguous in reference.")
    elif len(index) == 0:
        raise ValueError("Sentinel is not in reference.")

    return index[0], df


def greedy_select(df, reference):
    ld_threshold = 0.5

    selected = []
    cur = df.index[0]
    while True:
        chrom, pos = df.loc[cur, ["chrom", "pos"]]

        selected.append(cur)

        ###
        sentinel_index, ld_candidates = get_ld_candidates(df, chrom, pos)

        # Compute LD with LD candidates.
        bim = reference.bim
        index = []
        genotypes = []
        for idx, candidate in ld_candidates.iterrows():
            ref_snps = bim.loc[
                (bim["chrom"] == candidate.chrom) &
                (bim["pos"] == candidate.pos),
                ["snp", "chrom", "pos"]
            ]

            # Because some positions can have multiple entries, we remember
            # the point where we add the sentinel variant.
            if idx == sentinel_index:
                sentinel_index = len(index)

            # There might be multiple SNPs at a given position in the
            # reference.
            for name, row in ref_snps.iterrows():
                index.append((name, row.chrom, row.pos))
                genotypes.append(
                    reference.get_genotypes(name).genotypes.values[:, 0]
                )

        genotypes = np.array(genotypes).T
        # TODO This computes all the pairs and we only need a single row.
        ld_mat = ld(genotypes) ** 2

        # Get the correlated positions.
        correlated_indices = np.where(ld_mat[sentinel_index, :] > ld_threshold)
        print(correlated_indices)
        break

        cur = df.index[0]


def main():
    columns = {
        "chrom": "chrom",
        "pos": "pos",
        "effect_allele": "a1",
        "reference_allele": "a2",
        "coefficient": "beta",
        "significance": "pvalue"
    }

    stats = parse_summary_statistics(
        "/Users/legaultmarc/projects/StatGen/hdl_grs/data/summary.txt",
        columns,
        "\t"
    )

    stats = apply_significance_threshold(stats, p=5e-6)

    reference = get_reference(
        "/Users/legaultmarc/projects/StatGen/grs/test_data/big"
    )

    choices = greedy_select(stats, reference)
