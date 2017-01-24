"""
Multiple utilities to manipulate computed GRS.
"""

import logging
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


plt.style.use("ggplot")


def _read_grs(filename):
    return pd.read_csv(filename, sep=",", index_col="sample")


def histogram(args):
    out = args.out if args.out else "grs_histogram.png"
    data = _read_grs(args.grs_filename)

    plt.hist(data["grs"], bins=args.bins)
    logger.info("WRITING histogram to file '{}'.".format(out))
    if args.out.endswith(".png"):
        plt.savefig(out, dpi=300)
    else:
        plt.savefig(out)


def quantiles(args):
    out = args.out if args.out else "grs_discretized.csv"
    data = _read_grs(args.grs_filename)

    q = float(args.k) / args.q
    low, high = data.quantile([q, 1-q]).values.T[0]

    data["group"] = np.nan
    data.loc[data["grs"] <= low, "group"] = 0
    data.loc[data["grs"] > high, "group"] = 1

    if not args.keep_unclassified:
        data = data.dropna(axis=0, subset=["group"])

    logger.info("WRITING discretized GRS using k={}; q={} to file '{}'."
                "".format(args.k, args.q, out))

    data[["group"]].to_csv(out)


def main():
    args = parse_args()

    command_handlers = {
        "histogram": histogram,
        "quantiles": quantiles,
    }

    command_handlers[args.command](args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Utilities to manipulated computed GRS."
    )

    parent = argparse.ArgumentParser(add_help=False)

    subparser = parser.add_subparsers(
        dest="command",
    )

    subparser.required = True

    parser.add_argument(
        "grs_filename",
        help="Path to the file containing the computed GRS."
    )

    parser.add_argument(
        "--out", "-o",
        default=None
    )

    # Histogram
    histogram_parse = subparser.add_parser(
        "histogram",
        help="Plot the histogram of the computed GRS.",
        parents=[parent]
    )

    histogram_parse.add_argument(
        "--bins",
        type=int,
        default=60
    )

    # Quantiles
    quantiles = subparser.add_parser(
        "quantiles",
        help=(
            "Dichotomize the GRS using quantiles. Takes two parameters: "
            "k and q where q is the number of quantiles and k is the cutoff "
            "to be used for the discretization. For example, if the 1st "
            "quintile is to be compared to the 5th, use -q 5 -k 1. "
            "By default -k=1 and -q=2 which means the median is used."
        ),
        parents=[parent]
    )

    quantiles.add_argument("-k", default=1, type=int)
    quantiles.add_argument("-q", default=2, type=int)
    quantiles.add_argument("--keep-unclassified", action="store_true")

    return parser.parse_args()
