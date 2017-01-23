"""
Utility to match lists of SNPs and to find tags if needed.
"""


import sqlite3
import logging
import argparse
import collections

import numpy as np
from gepyto.structures.region import Region
from genetest.genotypes import format_map
from genetest.genotypes.core import Representation


Locus = collections.namedtuple("Locus", ("chrom", "pos"))

logger = logging.getLogger(__name__)


class Results(object):
    def __init__(self, filename, header):
        # Open the output file.
        self.f = open(filename, "w")
        self.sep = ","
        self.f.write(self.sep.join(header) + "\n")

    def write(self, *args):
        self.f.write(
            self.sep.join([str(i) for i in args]) + "\n"
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


def match_name(results, warn, cur):
    """Match variant by name in the source and target lists."""
    cur.execute(
        "SELECT s.name, t.name, COUNT(s.name) "
        "FROM source s, target t "
        "WHERE"
        "  s.name=t.name AND"
        "  s.name NOT IN (SELECT source FROM matches) "
        "GROUP BY s.name"
    )

    inserts = []
    for src, tar, n in cur:
        if n > 1:
            warn.write(
                src,
                "NAME_MATCH",
                "{} matches by name (ambiguous)".format(n)
            )
        else:
            inserts.append((src, tar, "NAME_MATCH"))

    cur.executemany("INSERT INTO matches VALUES (?, ?, ?)", inserts)


def match_variant(results, warn, cur):
    """Match variant by chromosome position and alleles.

    Dots can be used to match any allele.

    FIXME this can be pretty slow.

    """
    # First, match by locus without looking at the alleles.
    cur.execute(
        "SELECT s.name, t.name, s.chrom, s.pos, "
        "  lower(s.a1), lower(s.a2), lower(t.a1), lower(t.a2) "
        "FROM source s, target t "
        "WHERE "
        "  s.chrom=t.chrom AND s.pos=t.pos AND"
        "  s.name NOT IN (SELECT source FROM matches)"
    )

    inserts = []
    for src, tar, chrom, pos, s_a1, s_a2, t_a1, t_a2 in cur:
        # Check that alleles match and insert the match.
        insert = False
        alleles = _extend_with_complement({s_a1, s_a2})
        if {t_a1, t_a2} <= alleles:
            insert = True
        elif "." in {s_a1, s_a2} or "." in {t_a1, t_a2}:
            insert = True

        if insert:
            inserts.append([src, tar, "VARIANT_MATCH"])

    cur.executemany("INSERT INTO matches VALUES (?, ?, ?)", inserts)


def find_tags(cur, reference, reference_format, prefix=""):
    """Find tags in the reference dataset."""
    LD_WINDOW = 100e3  # 100kb

    regions = {}

    # Create a genomic range that covers all the SNPs that are missing from
    # the dataset.
    # This is done by padding every SNP position with LD_WINDOW and
    # concatenating the individual loci.
    logger.info("Building genomic regions for reference SNP extraction...")
    cur.execute(
        "SELECT chrom, pos "
        "FROM source "
        "WHERE source.name NOT IN (SELECT source FROM matches)"
    )

    for chrom, pos in cur:
        region = Region(
            chrom,
            max(0, pos - LD_WINDOW // 2),
            pos + LD_WINDOW // 2,
        )

        if regions.get(chrom) is None:
            regions[chrom] = region
        else:
            regions[chrom] = regions[chrom].union(region)

    # Extract genotypes in every region.
    if reference_format not in format_map.keys():
        raise ValueError(
            "Unknown reference format '{}'. Must be a genetest compatible "
            "format ({}).".format(reference_format, list(format_map.keys()))
        )

    container = format_map[reference_format](reference,
                                             Representation.ADDITIVE)

    logger.info("Reading genotypes from the reference...")
    genotypes = {}
    for snp in container.iter_marker_genotypes():
        chrom = str(snp.chrom)
        region = regions.get(chrom)
        if region and Locus(chrom, snp.pos) in region:
            # Add the genotypes.
            if genotypes.get(chrom) is None:
                genotypes[chrom] = snp.genotypes
                genotypes[chrom].columns = [snp.marker]

            else:
                genotypes[chrom][snp.marker] = snp.genotypes.values[:, 0]

    logger.info("Computing LD...")
    # Compute the pairwise LD matrices for every chromosome.
    for chrom, g in genotypes.items():
        mat = g.values
        mat = (mat - np.mean(mat, axis=0)) / np.std(mat, axis=0)
        r = np.dot(mat.T, mat) / mat.shape[0]
        np.save("{}ld.chr{}.npy".format(prefix, chrom), r)

    # Find the available variant in highest LD with the missing variants in
    # the source file.
    # TODO


def main():
    args = parse_args()

    prefix = "{}.".format(args.out) if args.out else ""

    if args.memory:
        db_filename = ":memory:"
    else:
        db_filename = "{}matcher.db".format(prefix)

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    def do_inserts(rows, table, cur):
        cur.executemany(
            "INSERT INTO {} VALUES (?, ?, ?, ?, ?)".format(table), rows
        )

    for table in ("source", "target"):
        cur.execute(
            "CREATE TABLE {} ("
            "  name TEXT NOT NULL,"
            "  chrom TEXT NOT NULL,"
            "  pos INTEGER NOT NULL,"
            "  a1 TEXT,"
            "  a2 TEXT"
            ")".format(table)
        )

        with open(getattr(args, table), "r") as f:
            header = f.readline()
            assert (
                header.strip().split(",") == ["name", "chrom", "pos", "a1", "a2"]
            )
            buf = []
            for line in f:
                line = [
                    i if i != "" else None for i in line.strip().split(",")
                ]
                assert len(line) == 5

                buf.append(line)
                if len(buf) > 5000:
                    do_inserts(buf, table, cur)
                    buf = []

            if buf:
                do_inserts(buf, table, cur)

    cur.execute(
        "CREATE TABLE matches ("
        "  source TEXT,"
        "  target TEXT,"
        "  how TEXT,"
        "  FOREIGN KEY(source) REFERENCES source(name),"
        "  FOREIGN KEY(target) REFERENCES target(name)"
        ")"
    )

    con.commit()

    # n the number of variants in the source.
    cur.execute("SELECT COUNT(*) FROM source")
    n = cur.fetchone()[0]

    results_cols = ["source_name", "target_name", "method"]
    warn_cols = ["source_name", "method", "message"]

    logger.info("Matching variants...")

    out_filename = "{}matcher.output.txt".format(prefix)
    warn_filename = "{}matcher.warning.txt".format(prefix)

    with Results(out_filename, results_cols) as out, \
         Results(warn_filename, warn_cols) as warn:
        for matcher in MATCHERS:
            matcher(out, warn, cur)

        cur.execute("SELECT * FROM matches")
        for src, tar, method in cur:
            out.write(src, tar, method)

    # Check number of matches.
    cur.execute("SELECT COUNT(*) FROM matches")
    n_matches = cur.fetchone()[0]

    logger.info(
        "Done matching variants. Found hits for {} / {}.".format(n_matches, n)
    )

    if args.reference and n_matches < n:
        logger.info("Finding tags in the reference for missing variants...")
        find_tags(cur, args.reference, args.reference_format, prefix)

    con.commit()
    con.close()


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

    parser.add_argument(
        "--memory",
        help="Load the matching database in memory instead of writing it.",
        action="store_true"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output prefix."
    )

    return parser.parse_args()


MATCHERS = [match_name, match_variant]
