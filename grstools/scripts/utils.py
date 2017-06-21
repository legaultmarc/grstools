"""
Multiple utilities to manipulate computed GRS.
"""

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
from genetest.subscribers import ResultsMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


plt.style.use("ggplot")
matplotlib.rc("font", size="6")


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


def beta_coefficient(args):
    # Get variants from summary stats file
    stats_variant = []
    stats_df = pd.read_csv(args.variants)

    for idx, row in stats_df.iterrows():
        v = geneparse.Variant(row["name"],
                              row["chrom"],
                              row["pos"],
                              [row["reference"], row["risk"]])
        stats_variant.append(v)

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
    extractor = geneparse.Extractor(reader, variants=stats_variant)

    # MODELSPEC
    # Phenotype container
    phenotypes = TextPhenotypes(args.phenotypes_filename,
                                sample_column=args.phenotypes_sample_column,
                                field_separator=args.phenotypes_separator)

    if args.test == "linear":
        model = spec.ModelSpec(
            outcome=spec.phenotypes[args.phenotype],
            predictors=[spec.SNPs],
            test=lambda:
                model_map[args.test](condition_value_t=float("infinity")), )

    else:
        model = spec.ModelSpec(
            outcome=spec.phenotypes[args.phenotype],
            predictors=[spec.SNPs],
            test=args.test)

    results_sub = ResultsMemory()

    execute(phenotypes,
            extractor,
            model,
            subscribers=[results_sub], cpus=10)

    # Put results from beta computation in df
    df = pd.DataFrame(results_sub.results)
    df_snps = pd.DataFrame(list(df.SNPs))

    df_snps.chrom = df_snps.chrom.map(lambda x: str(x)[3:])
    df_snps.chrom = df_snps.chrom.astype('int64')

    # Combine results from beta computation with summary stats
    df_all = pd.merge(df_snps,
                      stats_df,
                      how='outer',
                      left_on=['chrom', 'pos'],
                      right_on=['chrom', 'pos'],
                      indicator=True)

    df_common = df_all.query('_merge == "both"')

    # Variants with no computed beta
    df_stats_only = df_all.query('_merge != "both"')
    for idx, row in df_stats_only.iterrows():
        logger.warning("No beta computed for variant on chromosome {} "
                       "at position {}.".format(row.chrom, row.pos))

    # Get same risk and reference alleles for summary and computed stats
    df_inverted = df_common.loc[df_common.major != df_common.reference]
    df_inverted.major = df_inverted.reference
    df_inverted.minor = df_inverted.risk
    df_inverted.coef = abs(df_inverted.coef)

    df_not_inverted = df_common.loc[df_common.major == df_common.reference]

    df_common_sameAlleles = pd.concat([df_inverted, df_not_inverted])

    # Plot computed beta and effect from summary stats
    plt.plot(df_common_sameAlleles.effect, df_common_sameAlleles.coef, 'ro')
    plt.xlabel('Summary statistics coefficients')
    plt.ylabel('Computed coefficients')

    if args.out.endswith(".png"):
        plt.savefig(args.out, dpi=300)
    else:
        plt.savefig(args.out)


def main():
    args = parse_args()

    command_handlers = {
        "histogram": histogram,
        "quantiles": quantiles,
        "standardize": standardize,
        "correlation": correlation,
        "beta_coefficient": beta_coefficient
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

    # Beta_coefficients
    beta_coefficient = subparser.add_parser(
        "beta_coefficient",
        help="Compute beta coefficients from given genotypes "
             "data and compare them with beta coefficients from the "
             "grs file."
    )

    beta_coefficient.add_argument(
        "--variants",
        help="File describing the selected variants for GRS. "
             "The file must be in grs format",
        type=str,
        required=True
    )

    beta_coefficient.add_argument(
        "--genotypes-filename",
        help="File containing genotype data.",
        type=str,
        required=True
    )

    beta_coefficient.add_argument(
        "--genotypes-format",
        help=("File format of the genotypes in the reference (default:"
              "%(default)s)."),
        default="plink"
    )

    beta_coefficient.add_argument(
        "--genotypes-kwargs",
        help="Keyword arguments to pass to the genotype container."
             "A string of the following format is expected: "
             "key1=value1,key2=value2..."
             "It is also possible to prefix the values by 'int:' or 'float:' "
             "to cast them before passing them to the constructor.",
        type=str
    )

    beta_coefficient.add_argument(
        "--phenotype",
        help="Column name of phenotype in phenotypes file",
        type=str,
        required=True
    )

    beta_coefficient.add_argument(
        "--phenotypes-filename",
        help="File containing phenotype data",
        type=str,
        required=True
    )

    beta_coefficient.add_argument(
        "--phenotypes-separator",
        help="Separator in the phenotypes file (default:"
             "%(default)s).",
        type=str,
        default=","
    )

    beta_coefficient.add_argument(
        "--phenotypes-sample-column",
        help=("Column name of target phenotype (default:"
              "%(default)s)."),
        type=str,
        default="sample"
    )

    beta_coefficient.add_argument(
        "--test",
        help="Test to perform for beta coefficients estimation",
        type=str,
        required=True,
        choices=["linear", "logistic"]
    )

    beta_coefficient.add_argument(
        "--out", "-o",
        help=("Output filename for beta coefficients (default:"
              "%(default)s)."),
        default="beta_coefficients_plot.png"
    )

    return parser.parse_args()
