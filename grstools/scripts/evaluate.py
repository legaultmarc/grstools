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

from genetest.statistics import model_map

from ..utils import regress as _regress
from ..utils import parse_computed_grs_file, _create_genetest_phenotypes


plt.style.use("ggplot")
matplotlib.rc("font", size="7")


def _parse_phenotypes(args):
    """Parse a phenotypes file given the arguments added by
    _add_phenotype_arguments.
    """
    return pd.read_csv(
        args.phenotypes_filename, index_col=args.phenotypes_sample_column,
        sep=args.phenotypes_separator
    )


def _parse_and_regress(args, formula):
    phenotypes = _create_genetest_phenotypes(
        args.grs_filename, args.phenotypes_filename,
        args.phenotypes_sample_column, args.phenotypes_separator
    )
    return _regress(formula, args.test, phenotypes)


def regress(args):
    # Do the regression.
    formula = "{} ~ grs".format(args.phenotype)
    stats = _parse_and_regress(args, formula)

    if args.no_plot:
        print(json.dumps(stats))
        return

    # Read the files.
    grs = parse_computed_grs_file(args.grs_filename)
    phenotypes = _parse_phenotypes(args)
    df = phenotypes.join(grs)

    df = df[[args.phenotype, "grs"]]
    df.columns = ("y", "grs")

    # Create the plot.
    if args.test == "linear":
        return _linear_regress_plot(df, stats, args.out)
    if args.test == "logistic":
        return _logistic_regress_plot(df, stats, args.out)
    else:
        raise ValueError()


def _get_dummy_artist():
    return Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0
    )


def _linear_regress_plot(df, stats, out):
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
    plt.legend(
        (_get_dummy_artist(), data_marker, line_marker),
        (
            r"$\beta={:.3g}\ ({:.3g}, {:.3g}),\ (p={:.4g},\ R^2={:.3g})$"
            "".format(stats["beta"], stats["CI"][0], stats["CI"][1],
                      stats["p-value"], stats["R2"]),
            "data",
            "$y = {:.3g}grs + {:.3g}$"
            "".format(stats["beta"], stats["intercept"])
        )
    )

    if out is None:
        plt.show()
    else:
        plt.savefig(out)


def _logistic_regress_plot(df, stats, out):
    odds_ratio = np.exp(stats["beta"])
    odds_ratio_ci = [np.exp(i) for i in stats["CI"]]

    # Add the odd ratio or else the stats are for nothing.
    artists = [_get_dummy_artist()]
    labels = [
        "OR={:.3f} ({:.3f}, {:.3f}) (p={:.3g})"
        "".format(odds_ratio, odds_ratio_ci[0], odds_ratio_ci[1],
                  stats["p-value"])
    ]
    levels = df["y"].unique()
    boxplot_data = []
    for i, level in enumerate(levels):
        data = df.loc[df["y"] == level, "grs"]
        boxplot_data.append(data)

        noise = (np.random.random(data.shape[0]) - 0.5) / 4
        lines, = plt.plot(
            np.full_like(data, i + 1) + noise,
            data,
            "o",
            markersize=0.5,
        )
        artists.append(lines)
        labels.append(
            "GRS mean = {:.4f} ($\sigma={:.4f}$)"
            "".format(data.mean(), data.std())
        )

    plt.boxplot(boxplot_data, showfliers=False, medianprops={"color": "black"})

    plt.xlabel("Phenotype level")
    plt.xticks(range(1, len(levels) + 1), levels)

    plt.ylabel("GRS")

    plt.legend(artists, labels)

    if out is None:
        plt.show()
    else:
        plt.savefig(out)


def dichotomize_plot(args):
    """Compares differente quantiles of dichotomization."""
    # Read the files.
    grs = parse_computed_grs_file(args.grs_filename)
    phenotypes = _parse_phenotypes(args)
    df = phenotypes.join(grs)
    df["group"] = np.nan

    # Init the statistical test.
    test = model_map[args.test]()

    qs = []
    upper_ci = []
    lower_ci = []
    ns = []
    betas = []

    for q in np.linspace(0.05, 0.5, 200):
        low, high = df[["grs"]].quantile([q, 1 - q]).values.T[0]

        df["group"] = np.nan
        df.loc[df["grs"] <= low, "group"] = 0
        df.loc[df["grs"] >= high, "group"] = 1

        cur = df.dropna()

        stats = test.fit(
            cur[[args.phenotype]], cur[["group"]]
        )

        qs.append(q)
        betas.append(stats["group"]["coef"])
        ns.append(df.dropna().shape[0])
        upper_ci.append(stats["group"]["upper_ci"])
        lower_ci.append(stats["group"]["lower_ci"])

    fig, ax1 = plt.subplots()

    beta_line, = ax1.plot(qs, betas)
    ci_line, = ax1.plot(qs, upper_ci, "--", color="gray", linewidth=0.2)
    ax1.plot(qs, lower_ci, "--", color="gray", linewidth=0.2)
    ax1.set_ylabel(r"$\beta$")
    ax1.set_xlabel("Quantile used to form groups (0.5 is median)")

    ax2 = ax1.twinx()
    ax2.grid(False, which="both")
    n_line, = ax2.plot(qs, ns, "-", linewidth=0.2)
    ax2.set_ylabel("effective n")

    plt.legend(
        (beta_line, ci_line, n_line),
        (r"$\beta$", "95% CI", "$n$"),
        loc="upper center"
    )

    if args.out:
        plt.savefig(args.out)
    else:
        plt.show()


def main():
    args = parse_args()

    command_handlers = {
        "regress": regress,
        "dichotomize-plot": dichotomize_plot,
    }

    command_handlers[args.command](args)


def _add_phenotype_arguments(parser):
    parser.add_argument("--phenotypes-filename", type=str)
    parser.add_argument("--phenotypes-sample-column", type=str,
                        default="sample")
    parser.add_argument("--phenotypes-separator", type=str,
                        default=",")
    parser.add_argument("--phenotype", type=str)


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

    parent.add_argument("--out", "-o", default=None, type=str)

    subparser = parser.add_subparsers(
        dest="command",
    )

    subparser.required = True

    # Regress
    # TODO
    # To evaluate the performance of discretized GRS, it might be interesting
    # to generate simular plots of y ~ GRS. Then it could be qregress for
    # continuous GRS and dregress for discretized GRS.
    regress_parse = subparser.add_parser(
        "regress",
        help="Regress the GRS on a discrete or continuous outcome.",
        parents=[parent]
    )

    _add_phenotype_arguments(regress_parse)
    regress_parse.add_argument("--test", type=str)
    regress_parse.add_argument("--no-plot", action="store_true")

    # Dichotomize plot.
    dichotomize_parse = subparser.add_parser(
        "dichotomize-plot",
        help="A plot to help identify ideal dichotmizatin parameters.",
        parents=[parent]
    )

    _add_phenotype_arguments(dichotomize_parse)
    dichotomize_parse.add_argument("--test", type=str)

    return parser.parse_args()
