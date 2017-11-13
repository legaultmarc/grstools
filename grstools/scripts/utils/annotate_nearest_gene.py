"""
Annotate the GRS with the nearest gene.
"""


import sqlite3
import gzip
import csv


DB_COLS = ("ensembl_id", "chrom", "start", "end", "gene_name", "gene_biotype")


class MemoryDB(object):
    def __init__(self):
        self.con = sqlite3.connect(":memory:")
        self.cur = self.con.cursor()

    def close(self):
        self.con.close()


def annotate_nearest_gene(args):
    # Read the gene file.
    # Returns a MemoryDB
    # genes (ensembl_id, chrom, start, end, gene_name, gene_biotype)
    db = read_gtf(args.gene_reference_gtf)

    annotate_grs(args.grs_filename, db, out=args.out)


def annotate_grs(filename, db, out):
    """Write the annotated GRS file.

    The format will be the same as the GRS format with additional columns:

        - ensembl_id
        - gene_name
        - start
        - end
        - gene_biotype
        - distance (0 if overlapping)

    If multiple genes overlap a variant, there will be multiple rows for that
    variant.

    """
    output_filename = out
    if output_filename is None:
        output_filename = (
            filename.rstrip(".grs") + "_nearest_gene_annotated.csv"
        )

    with open(filename, "r") as grs, \
         open(output_filename, "w") as f:

        out = csv.writer(f, dialect="unix")
        out.writerow([
            "name", "chrom", "pos", "reference", "risk", "p-value", "effect",
            "ensembl_id", "gene_name", "gene_start", "gene_end", "gene_biotype",
            "distance"
        ])

        # Read the header.
        h = {k: i for i, k in enumerate(next(grs).strip().split(","))}

        for line in grs:
            line = line.strip().split(",")

            chrom = line[h["chrom"]]
            pos = int(line[h["pos"]])

            # Find if it overlaps a gene.
            db.cur.execute(
                "SELECT * FROM gene"
                "  WHERE chrom=:chrom AND start <= :pos AND :pos <= end",
                {"chrom": chrom, "pos": pos}
            )
            matching_genes = db.cur.fetchall()

            if len(matching_genes) > 0:
                for tu in matching_genes:
                    out.writerow(format_row(line, h, tu, 0))
                continue

            # Find the nearest gene (there was no gene directly overlapping.
            db.cur.execute(
                "SELECT * FROM gene"
                "  WHERE chrom=:chrom"
                "  ORDER BY MIN(ABS(:pos - start), ABS(:pos - end))"
                "  LIMIT 1",
                {"chrom": chrom, "pos": pos}
            )
            closest = db.cur.fetchone()
            ens_start = closest[DB_COLS.index("start")]
            ens_end = closest[DB_COLS.index("end")]
            distance = min(abs(pos - ens_start), abs(pos - ens_end))

            out.writerow(format_row(line, h, closest, distance))


def format_row(line, h, tu, distance=0):
    """Format an output row given a GRS line and a gene match tuple."""
    assert len(tu) == len(DB_COLS)
    hit = {col: val for col, val in zip(DB_COLS, tu)}

    return [
        line[h["name"]], line[h["chrom"]], line[h["pos"]],
        line[h["reference"]], line[h["risk"]],
        line[h["p-value"]], line[h["effect"]],
        hit["ensembl_id"], hit["gene_name"],
        hit["start"], hit["end"], hit["gene_biotype"], distance
    ]


def read_gtf(filename):
    db = _initialize_db()

    buf = []
    with gzip.open(filename, "rt") as f:
        for line in f:
            if line.startswith("#!"):
                continue

            # Unused columns:
            # source = line[1]
            # score = line[5]
            # strand = line[6]
            # frame = line[7]

            line = line.rstrip().split("\t")
            chrom = line[0]
            feature = line[2]
            start = line[3]
            end = line[4]

            attributes = _parse_attributes(line[8])

            if feature != "gene":
                continue

            if attributes.get("gene_biotype", "") != "protein_coding":
                continue

            buf.append((
                attributes.get("gene_id"),
                chrom,
                int(start),
                int(end),
                attributes.get("gene_name"),
                attributes.get("gene_biotype"),
            ))

    db.cur.executemany("INSERT INTO gene VALUES (?, ?, ?, ?, ?, ?)", buf)
    del buf

    return db


def _parse_attributes(field):
    d = {}
    for attribute in field.split(";"):
        attribute = attribute.strip()
        if attribute == "":
            continue

        if "=" in attribute:
            key, value = attribute.split("=")
        else:
            key, value = attribute.split(" ")

        value = value.strip("\"")
        d[key] = value

    return d


def _initialize_db():
    db = MemoryDB()
    cur = db.cur

    cur.execute(
        "CREATE TABLE gene ("
        "  ensembl_id TEXT PRIMARY KEY,"
        "  chrom TEXT,"
        "  start INT,"
        "  end INT,"
        "  gene_name TEXT,"
        "  gene_biotype TEXT"
        ")"
    )

    cur.execute(
        "CREATE INDEX _pos_idx ON gene (chrom, start, end)"
    )

    db.con.commit()

    return db
