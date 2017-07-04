"""
Multiple utilities to manipulate computed GRS.
"""

# This file is part of grstools.
#
# The MIT License (MIT)
#
# Copyright (c) 2017 Marc-Andre Legault
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import logging
import argparse

import pandas as pd
import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

import geneparse

from ..utils import parse_computed_grs_file

from genetest.phenotypes import TextPhenotypes

import genetest.modelspec as spec
from genetest.analysis import execute
from genetest.statistics import model_map
from genetest.subscribers import Subscriber

from multiprocessing import cpu_count

logger = logging.getLogger(__name__)


plt.style.use("ggplot")
matplotlib.rc("font", size="6")


class BetaTuple(object):
    __slots__ = ("e_risk", "e_coef", "e_error",
                 "o_risk", "o_coef", "o_error", "o_maf", "o_nobs")

    def __init__(self, e_risk, e_coef):
        # e:expected
        self.e_risk = e_risk
        self.e_coef = float(e_coef)
        self.e_error = None

        # o:observed (computed)
        self.o_risk = None
        self.o_coef = None
        self.o_error = None
        self.o_maf = None
        self.o_nobs = None


class BetaSubscriber(Subscriber):

    def __init__(self, variant_to_expected):
        self.variant_to_expected = variant_to_expected

    def handle(self, results):
        v = geneparse.Variant("None",
                              results["SNPs"]["chrom"],
                              results["SNPs"]["pos"],
                              [results["SNPs"]["major"],
                               results["SNPs"]["minor"]])

        # Same reference and risk alleles for expected and observed
        if self.variant_to_expected[v].e_risk == results["SNPs"]["minor"]:
            self.variant_to_expected[v].o_risk = results["SNPs"]["minor"]
            self.variant_to_expected[v].o_coef = results["SNPs"]["coef"]
            self.variant_to_expected[v].o_maf = results["SNPs"]["maf"]

        else:
            self.variant_to_expected[v].o_risk = results["SNPs"]["major"]
            self.variant_to_expected[v].o_coef = -results["SNPs"]["coef"]
            self.variant_to_expected[v].o_maf = 1 - results["SNPs"]["maf"]

        self.variant_to_expected[v].o_error = results["SNPs"]["std_err"]
        self.variant_to_expected[v].o_nobs = results["MODEL"]["nobs"]


def histogram(args):
    out = args.out if args.out else "grs_histogram.png"
    data = parse_computed_grs_file(args.grs_filename)

    plt.hist(data["grs"], bins=args.bins)
    plt.xlabel("GRS")
    logger.info("WRITING histogram to file '{}'.".format(out))

    if out.endswith(".png"):
        plt.savefig(out, dpi=300)
    else:
        plt.savefig(out)


def quantiles(args):
    out = args.out if args.out else "grs_discretized.csv"
    data = parse_computed_grs_file(args.grs_filename)

    q = float(args.k) / args.q
    low, high = data.quantile([q, 1-q]).values.T[0]

    data["group"] = np.nan
    data.loc[data["grs"] <= low, "group"] = 0
    data.loc[data["grs"] >= high, "group"] = 1

    if not args.keep_unclassified:
        data = data.dropna(axis=0, subset=["group"])

    logger.info("WRITING discretized GRS using k={}; q={} to file '{}'."
                "".format(args.k, args.q, out))

    data[["group"]].to_csv(out)


def standardize(args):
    out = args.out if args.out else "grs_standardized.csv"
    data = parse_computed_grs_file(args.grs_filename)

    data["grs"] = (data["grs"] - data["grs"].mean()) / data["grs"].std()
    data.to_csv(out)


def correlation(args):
    grs1 = parse_computed_grs_file(args.grs_filename)
    grs1.columns = ["grs1"]

    grs2 = parse_computed_grs_file(args.grs_filename2)
    grs2.columns = ["grs2"]

    grs = pd.merge(grs1, grs2, left_index=True, right_index=True, how="inner")

    if grs.shape[0] == 0:
        raise ValueError("No overlapping samples between the two GRS.")

    linreg = scipy.stats.linregress
    slope, intercept, r_value, p_value, std_err = linreg(grs["grs1"],
                                                         grs["grs2"])

    plt.scatter(grs["grs1"], grs["grs2"], marker=".", s=1, c="#444444",
                label="data")

    xmin = np.min(grs["grs1"])
    xmax = np.max(grs["grs1"])

    x = np.linspace(xmin, xmax, 2000)

    plt.plot(
        x, slope * x + intercept,
        label=("GRS2 = {:.2f} GRS1 + {:.2f} ($R^2={:.2f}$)"
               "".format(slope, intercept, r_value ** 2)),
        linewidth=0.5
    )

    plt.plot(x, x, label="GRS2 = GRS1", linestyle="--", linewidth=0.5,
             color="#777777")

    plt.xlabel("GRS1")
    plt.ylabel("GRS2")

    plt.legend()
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out)
    else:
        plt.show()


def beta_plot(args):
    # Key:Variant instance, value: beta_tuple instance
    variant_to_expected = {}

    # Get variants from summary stats file
    with open(args.summary, "r") as f:
        header = f.readline()
        header_to_pos = {title: pos for pos, title in enumerate(
            header.strip().split(","))}

        expected_headers = {"name", "chrom", "pos", "reference",
                            "risk", "p-value", "effect"}

        missing_headers = expected_headers - header_to_pos.keys()

        if len(missing_headers) != 0:
            raise ValueError(
                "Missing the columns {} in variants input file".format(
                    ",".join(missing_headers)
                )
            )

        for line in f:
            l = line.split(",")
            v = geneparse.Variant(
                None,
                l[header_to_pos["chrom"]],
                l[header_to_pos["pos"]],
                [l[header_to_pos["reference"]],
                 l[header_to_pos["risk"]]]
            )

            variant_to_expected[v] = BetaTuple(
                l[header_to_pos["risk"]],
                l[header_to_pos["effect"]]
            )

    # Extract genotypes
    if args.genotypes_format not in geneparse.parsers.keys():
        raise ValueError(
            "Unknown reference format '{}'. Must be a genetest compatible "
            "format ({})."
            "".format(args.genotypes_format, list(geneparse.parsers.keys()))
        )

    genotypes_kwargs = {}
    if args.genotypes_kwargs:
        for argument in args.genotypes_kwargs.split(","):
            key, value = argument.strip().split("=")

            if value.startswith("int:"):
                value = int(value[4:])

            elif value.startswith("float:"):
                value = float(value[6:])

            genotypes_kwargs[key] = value

    reader = geneparse.parsers[args.genotypes_format]
    reader = reader(
        args.genotypes_filename,
        **genotypes_kwargs
    )

    extractor = geneparse.Extractor(reader,
                                    variants=variant_to_expected.keys())

    # MODELSPEC
    # Phenotype container
    phenotypes = TextPhenotypes(args.phenotypes_filename,
                                sample_column=args.phenotypes_sample_column,
                                field_separator=args.phenotypes_separator)

    # Test
    if args.test == "linear":
        def test_specification():
            return model_map[args.test](
                condition_value_t=float("infinity")
            )

    else:
        test_specification = args.test

    # Covariates
    pred = [spec.SNPs]
    if args.covar is not None:
        pred.extend(
            [spec.phenotypes[c] for c in args.covar.split(",")]
        )

    # Model
    model = spec.ModelSpec(
        outcome=spec.phenotypes[args.phenotype],
        predictors=pred,
        test=test_specification)

    # Subscriber
    custom_sub = BetaSubscriber(variant_to_expected)

    # Execution
    execute(phenotypes,
            extractor,
            model,
            subscribers=[custom_sub], cpus=args.cpus)

    # Plot and write to file observed and expected beta coefficients
    xs = []

    ys = []
    ys_error = []

    f = open(args.out + ".txt", "w")
    f.write("chrom,position,alleles,risk,expected_coef,"
            "observed_coef,observed_se,observed_maf,n\n")

    for variant, statistic in variant_to_expected.items():
        if statistic.o_coef is None:
            logger.warning("No statistic for {}".format(variant))

        else:
            # Plot
            xs.append(statistic.e_coef)
            ys.append(statistic.o_coef)

            if not args.no_error_bars:
                ys_error.append(statistic.o_error)

            # File
            line = [str(variant.chrom), str(variant.pos),
                    "/".join(variant.alleles_set),
                    statistic.e_risk, str(statistic.e_coef),
                    str(statistic.o_coef), str(statistic.o_error),
                    str(statistic.o_maf), str(statistic.o_nobs)]
            line = ",".join(line)
            f.write(line + "\n")

    f.close()

    if not args.no_error_bars:
        plt.errorbar(xs, ys, yerr=ys_error, fmt='.', markersize=3, capsize=2,
                     markeredgewidth=0.5, elinewidth=0.5, ecolor='black')
    else:
        plt.plot(xs, ys, '.', markersize=3)

    plt.xlabel('Expected coefficients')
    plt.ylabel('Observed coefficients')

    plt.savefig(args.out + ".png")


def main():
    args = parse_args()

    command_handlers = {
        "histogram": histogram,
        "quantiles": quantiles,
        "standardize": standardize,
        "correlation": correlation,
        "beta-plot": beta_plot,
    }

    command_handlers[args.command](args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Utilities to manipulate computed GRS."
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

    # Histogram
    histogram_parse = subparser.add_parser(
        "histogram",
        help="Plot the histogram of the computed GRS.",
        parents=[parent]
    )

    histogram_parse.add_argument("--bins", type=int, default=60)

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

    # Standardize
    subparser.add_parser(
        "standardize",
        help="Standardize the GRS (grs <- (grs - mean) / std).",
        parents=[parent]
    )

    # Correlation
    correlation = subparser.add_parser(
        "correlation",
        help="Plot the correlation between two GRS.",
        parents=[parent]
    )

    correlation.add_argument(
        "grs_filename2",
        help="Filename of the second GRS."
    )

    # Beta_plot
    beta_plot = subparser.add_parser(
        "beta-plot",
        help="Compute beta coefficients from given genotypes "
             "data and compare them with beta coefficients from the "
             "grs file."
    )

    beta_plot.add_argument(
        "--summary",
        help="File describing the selected variants for GRS. "
             "The file must be in grs format",
        type=str,
        required=True
    )

    beta_plot.add_argument(
        "--genotypes-filename",
        help="File containing genotype data.",
        type=str,
        required=True
    )

    beta_plot.add_argument(
        "--genotypes-format",
        help=("File format of the genotypes in the reference (default:"
              "%(default)s)."),
        default="plink"
    )

    beta_plot.add_argument(
        "--genotypes-kwargs",
        help="Keyword arguments to pass to the genotype container."
             "A string of the following format is expected: "
             "key1=value1,key2=value2..."
             "It is also possible to prefix the values by 'int:' or 'float:' "
             "to cast them before passing them to the constructor.",
        type=str
    )

    beta_plot.add_argument(
        "--phenotype",
        help="Column name of phenotype in phenotypes file",
        type=str,
        required=True
    )

    beta_plot.add_argument(
        "--phenotypes-filename",
        help="File containing phenotype data",
        type=str,
        required=True
    )

    beta_plot.add_argument(
        "--phenotypes-separator",
        help="Separator in the phenotypes file (default:"
             "%(default)s).",
        type=str,
        default=","
    )

    beta_plot.add_argument(
        "--phenotypes-sample-column",
        help=("Column name of target phenotype (default:"
              "%(default)s)."),
        type=str,
        default="sample"
    )

    beta_plot.add_argument(
        "--test",
        help="Test to perform for beta coefficients estimation",
        type=str,
        required=True,
        choices=["linear", "logistic"]
    )

    beta_plot.add_argument(
        "--no-error-bars",
        help="Do not show error bars on the plot",
        action="store_true"
    )

    beta_plot.add_argument(
        "--out", "-o",
        help=("Output name for beta coefficients plot and file (default:"
              "%(default)s)."),
        default="observed_coefficients"
    )

    beta_plot.add_argument(
        "--cpus",
        help=("Number of cpus to use for execution (default: "
              "number of cpus - 1 = %(default)s)."),
        type=int,
        default=max(cpu_count() - 1, 1)
    )

    beta_plot.add_argument(
        "--covar",
        help=("Covariates other than the SNPs from summary stats. "
              "Covariates should be in string form, separated by , "
              "covar1,covar2,covar3...")
    )

    return parser.parse_args()
