"""
Main function for the grs-utils script.
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

from multiprocessing import cpu_count
import argparse

import matplotlib
import matplotlib.pyplot as plt

from . import (histogram,
               quantiles,
               standardize,
               correlation,
               beta_plot,
               annotate_nearest_gene,
               locus_overlap)


plt.style.use("ggplot")
matplotlib.rc("font", size="6")


def main():
    args = parse_args()

    command_handlers = {
        "histogram": histogram,
        "quantiles": quantiles,
        "standardize": standardize,
        "correlation": correlation,
        "beta-plot": beta_plot,
        "annotate-nearest-gene": annotate_nearest_gene,
        "locus-overlap": locus_overlap,
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
        "--svg",
        help="Use svg format for output plot (else will be .png)",
        action="store_true"
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

    # annotate_nearest_gene
    annotate_nearest_gene = subparser.add_parser(
        "annotate-nearest-gene",
        help="Annotate a GRS with the nearest gene for each SNP. "
             "This script relies on an external gene annotation file such "
             "as the GTF file provided by Ensembl.",
        parents=[parent],
    )

    annotate_nearest_gene.add_argument(
        "--gene-reference-gtf",
        help="Gene annotation file. The script was tested with "
             "ftp://ftp.ensembl.org/pub/grch37/update/gtf/homo_sapiens/Homo_sapiens.GRCh37.87.gtf.gz"
    )

    # Locus_overlap
    locus_overlap = subparser.add_parser(
        "locus-overlap",
        help="Display variants in LD."
    )

    locus_overlap.add_argument(
        "--variants",
        help="File(s) describing the variants in grs format. "
             "The format should be name1=filename1,name2=filename2,...,"
             "nameN=filenameN where name1,...nameN are names to identify "
             "from which file variants come from.",
        type=str,
        required=True
    )

    locus_overlap.add_argument(
        "--genotypes-filename",
        help="File containing genotype data.",
        type=str,
        required=True
    )

    locus_overlap.add_argument(
        "--genotypes-format",
        help=("File format of the genotypes in the reference (default:"
              "%(default)s)."),
        default="plink"
    )

    locus_overlap.add_argument(
        "--genotypes-kwargs",
        help="Keyword arguments to pass to the genotype container."
             "A string of the following format is expected: "
             "key1=value1,key2=value2..."
             "It is also possible to prefix the values by 'int:' or 'float:' "
             "to cast them before passing them to the constructor.",
        type=str,
    )

    locus_overlap.add_argument(
        "--ld-threshold",
        help="LD threshold. All variants in LD will be considered as "
        "belonging to the same locus.(default:%(default)s).)",
        default=0.1,
        type=float
    )

    return parser.parse_args()
