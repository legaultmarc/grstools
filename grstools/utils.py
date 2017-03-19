"""
Utilities to manage files.
"""

import logging

import pandas as pd


logger = logging.getLogger(__name__)


COL_TYPES = {
    "name": str, "chrom": str, "pos": int, "reference": str, "risk": str,
    "p-value": float, "effect": float
}


def parse_grs_file(filename, p_threshold=1, maf_threshold=0, sep=",",
                   log=False):
    """Parse a GRS file.

    The mandatory columns are:
        - name (variant name, needs to be unique and defined)
        - chrom (chromosome, a str))
        - pos (position, a int)
        - reference (reference allele)
        - risk (effect/risk allele)
        - p-value (p-value, a float)
        - effect (beta or OR or other form of weight, a float)

    Optional columns are:
        - maf

    Returns:
        A pandas dataframe.

    """
    df = pd.read_csv(filename, sep=sep, dtype=COL_TYPES)

    try:
        df.set_index("name", inplace=True, verify_integrity=True)
    except ValueError:
        raise ValueError("Variant names are not unique.")

    cols = list(COL_TYPES.keys())
    cols.remove("name")

    # Optional columns.
    if "maf" in df.columns:
        cols.append("maf")

    # This will raise a KeyError if needed.
    df = df[cols]

    # Make the alleles uppercase.
    df["reference"] = df["reference"].str.upper()
    df["risk"] = df["risk"].str.upper()

    # Apply thresholds.
    if log:
        logger.info("Applying p-value threshold (p <= {})."
                    "".format(p_threshold))

    df = df.loc[df["p-value"] <= p_threshold, :]

    if "maf" in df.columns:
        if log:
            logger.info("Applying MAF threshold (MAF >= {})."
                        "".format(maf_threshold))
        df = df.loc[df["maf"] >= maf_threshold, :]

    return df
