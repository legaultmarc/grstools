"""
Evaluate the performance of constructed GRS or of the variant selection
procedure.
"""

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..utils import regress as _regress
from ..utils import parse_computed_grs_file


plt.style.use("ggplot")
matplotlib.rc("font", size="5")


def regress(args):
    # Do the regression.
    formula = "{} ~ grs".format(args.phenotype)
    stats = _regress(
        formula, args.test, args.grs_filename, args.phenotypes_filename,
        args.phenotypes_sample_column, args.phenotypes_separator
    )

    if args.no_plot:
        print(json.dumps(stats))
        return

    # Read the files.
    grs = parse_computed_grs_file(args.grs_filename)
    phenotypes = pd.read_csv(
        args.phenotypes_filename, index_col=args.phenotypes_sample_column,
        sep=args.phenotypes_separator
    )

    df = phenotypes.join(grs)

    df = df[[args.phenotype, "grs"]]
    df.columns = ("y", "grs")

    # Create the plot.
    if args.test == "linear":
        return _linear_regress_plot(df, stats)
    if args.test == "logistic":
        return _logistic_regress_plot(df, stats)
    else:
        raise ValueError()


def _linear_regress_plot(df, stats):
    data_marker, = plt.plot(df["grs"], df["y"], "o", markersize=0.5)

    xmin = df["grs"].min()
    xmax = df["grs"].max()

    line_marker, = plt.plot(
        [xmin, xmax],
        [stats["beta"] * xmin + stats["intercept"],
         stats["beta"] * xmax + stats["intercept"]],
        "-",
        linewidth=0.5,
        color="black",
    )

    plt.xlabel("GRS")
    plt.ylabel("Phenotype")

    # Add extra info to the legend.
    rect = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0
    )

    plt.legend(
        (rect, data_marker, line_marker),
        (
            r"$\beta={:.3g}\ ({:.3g}, {:.3g}),\ (p={:.4g},\ R^2={:.3g})$"
            "".format(stats["beta"], stats["CI"][0], stats["CI"][1],
                      stats["p-value"], stats["R2"]),
            "data",
            "$y = {:.3g}grs + {:.3g}$"
            "".format(stats["beta"], stats["intercept"])
        )
    )

    plt.show()


def _logistic_regress_plot(df, stats):
    levels = df["y"].unique()
    boxplot_data = []
    for i, level in enumerate(levels):
        data = df.loc[df["y"] == level, "grs"]
        boxplot_data.append(data)

        noise = (np.random.random(data.shape[0]) - 0.5) / 4
        plt.plot(
            np.full_like(data, i + 1) + noise,
            data,
            "o",
            markersize=0.5,
            label="GRS mean = {:.4f} ($\sigma={:.4f}$)".format(data.mean(),
                                                               data.std())
        )

    plt.boxplot(boxplot_data, showfliers=False, medianprops={"color": "black"})

    plt.xlabel("Phenotype level")
    plt.xticks(range(1, len(levels) + 1), levels)

    plt.ylabel("GRS")

    plt.legend()

    plt.show()


def main():
    args = parse_args()

    command_handlers = {
        "regress": regress,
    }

    command_handlers[args.command](args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Utilities to evaluate the performance of GRS."
    )

    parent = argparse.ArgumentParser(add_help=False)

    # General arguments.
    parent.add_argument(
        "grs_filename",
        help="Path to the file containing the computed GRS."
    )

    parent.add_argument(
        "--out", "-o",
        default=None
    )

    subparser = parser.add_subparsers(
        dest="command",
    )

    subparser.required = True

    # Regress
    regress_parse = subparser.add_parser(
        "regress",
        help="Regress the GRS on an outcome.",
        parents=[parent]
    )

    regress_parse.add_argument("--phenotypes-filename", type=str)
    regress_parse.add_argument("--phenotypes-sample-column", type=str,
                               default="sample")
    regress_parse.add_argument("--phenotypes-separator", type=str,
                               default=",")
    regress_parse.add_argument("--phenotype", type=str)
    regress_parse.add_argument("--test", type=str)
    regress_parse.add_argument("--no-plot", action="store_true")

    return parser.parse_args()
