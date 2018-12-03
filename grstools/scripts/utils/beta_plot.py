import logging

import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

import geneparse
import genetest.modelspec as spec
from genetest.phenotypes import TextPhenotypes
from genetest.analysis import execute
from genetest.statistics import model_map
from genetest.subscribers import Subscriber


logger = logging.getLogger(__name__)


class BetaTuple(object):
    """Data structure to hold information about the variants that are compared.

    """
    __slots__ = ("e_risk", "e_coef", "e_error",
                 "o_coef", "o_error", "o_maf", "o_nobs",
                 "valid", "message")

    def __init__(self, e_risk, e_coef, e_error):
        # e:expected
        self.e_risk = e_risk
        self.e_coef = float(e_coef)
        self.e_error = e_error

        # o:observed (computed)
        self.o_coef = None
        self.o_error = None
        self.o_maf = None
        self.o_nobs = None

        self.valid = False
        self.message = None


class BetaSubscriber(Subscriber):
    """Genetest subscriber. """
    def __init__(self, variant_to_expected):
        self.variant_to_expected = variant_to_expected
        self.variants_to_remove = set()

    def handle(self, results):
        v = geneparse.Variant(
            "None",
            results["SNPs"]["chrom"],
            results["SNPs"]["pos"],
            [results["SNPs"]["major"], results["SNPs"]["minor"]]
        )

        if results["SNPs"]["maf"] < 0.01:
            logger.warning("Ignoring {} because its maf ({}) is "
                           "less than 1%".format(v, results["SNPs"]["maf"]))

        # Get the variant in the information dictionary.
        info = self.variant_to_expected[v]

        # Variants to remove from results
        if results["SNPs"]["coef"] is None:
            info.message = "No statistic for {}".format(v)
            logger.warning(info.message)
            self.variants_to_remove.add(v)
            return

        elif results["SNPs"]["maf"] < 0.01:
            info.message = (
                "Ignoring {} because its maf ({}) is less than 1%."
                "".format(v, results["SNPs"]["maf"])
            )
            logger.warning(info.message)
            self.variant_to_remove.add(v)
            return

        # Same reference and risk alleles for expected and observed
        if info.e_risk == results["SNPs"]["minor"]:
            info.o_coef = results["SNPs"]["coef"]

        elif info.e_risk == results["SNPs"]["major"]:
            # Flip the coefficient as it is expressed wrt the wrong allele.
            info.o_coef = -results["SNPs"]["coef"]

        else:
            raise ValueError(
                "Unexpected allele {} for {}.".format(info.e_risk, v)
            )

        info.o_error = results["SNPs"]["std_err"]
        info.o_maf = results["SNPs"]["maf"]
        info.o_nobs = results["MODEL"]["nobs"]
        info.valid = True


def beta_plot(args):
    # Get variants from summary stats file
    variant_to_expected = get_summary_variants(args.summary)

    # Extract variants genotypes
    extractor = extract_variants_genotypes(
        args.genotypes_format,
        args.genotypes_kwargs,
        args.genotypes_filename,
        variant_to_expected
    )

    # Compute beta coefficients
    results = compute_beta_coefficients(
        args.phenotypes_filename,
        args.phenotype,
        args.phenotypes_sample_column,
        args.phenotypes_separator,
        args.covar,
        args.test,
        args.cpus,
        variant_to_expected,
        extractor
    )

    # Remove variants with no result found or with maf < 1%
    filter_results(variant_to_expected)

    # Create output file and plot
    create_outputs(
        args.out,
        args.svg,
        args.no_error_bars,
        results,
        variant_to_expected
    )


def get_summary_variants(summary_filename):
    """Read a summary statistics file in GRS format returning a dict of
    Variant to BetaTuple.

    """
    # Key:Variant instance, value: beta_tuple instance
    variant_to_expected = {}

    # Get variants from summary stats file
    with open(summary_filename, "r") as f:
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

            se = get_variant_standard_error(
                v,
                l[header_to_pos["p-value"]],
                l[header_to_pos["effect"]]
            )

            variant_to_expected[v] = BetaTuple(
                l[header_to_pos["risk"]],
                l[header_to_pos["effect"]],
                se
            )

        return variant_to_expected


def get_variant_standard_error(variant, pval, effect):
    return effect / abs(scipy.stats.norm.ppf(pval / 2.0))


def extract_variants_genotypes(genotypes_format, genotypes_kwargs,
                               genotypes_filename, variant_to_expected):
    # Extract genotypes
    if genotypes_format not in geneparse.parsers.keys():
        raise ValueError(
            "Unknown reference format '{}'. Must be a genetest compatible "
            "format ({})."
            "".format(genotypes_format, list(geneparse.parsers.keys()))
        )

    genotypes_kwargs_info = {}
    if genotypes_kwargs:
        for argument in genotypes_kwargs.split(","):
            key, value = argument.strip().split("=")

            if value.startswith("int:"):
                value = int(value[4:])

            elif value.startswith("float:"):
                value = float(value[6:])

            genotypes_kwargs_info[key] = value

    reader = geneparse.parsers[genotypes_format]
    reader = reader(genotypes_filename, **genotypes_kwargs_info)

    extractor = geneparse.Extractor(reader,
                                    variants=variant_to_expected.keys())

    return extractor


def compute_beta_coefficients(phenotypes_filename, phenotype,
                              phenotypes_sample_column, phenotypes_separator,
                              covar, test, nb_cpus, variant_to_expected,
                              extractor):
    # MODELSPEC
    # Phenotype container
    phenotypes = TextPhenotypes(phenotypes_filename,
                                sample_column=phenotypes_sample_column,
                                field_separator=phenotypes_separator)

    # Test
    if test == "linear":
        def test_specification():
            return model_map[test](
                condition_value_t=float("infinity")
            )

    else:
        test_specification = test

    # Covariates
    pred = [spec.SNPs]
    if covar is not None:
        pred.extend(
            [spec.phenotypes[c] for c in covar.split(",")]
        )

    # Model
    model = spec.ModelSpec(
        outcome=spec.phenotypes[phenotype],
        predictors=pred,
        test=test_specification
    )

    # Subscriber
    beta_sub = BetaSubscriber(variant_to_expected)

    # Execution
    execute(phenotypes,
            extractor,
            model,
            subscribers=[beta_sub], cpus=nb_cpus)

    return beta_sub


def filter_results(variant_to_expected):
    for variant, beta_tuple in variant_to_expected.items():
        if not beta_tuple.valid:
            if beta_tuple.message is None:
                beta_tuple.message = "No genotype found for {}".format(variant)


def create_outputs(out_filename, svg_format, no_error_bars,
                   beta_sub, variant_to_expected):
    # Plot and write to file observed and expected beta coefficients
    xs = []
    xs_error = []

    ys = []
    ys_error = []

    f = open(out_filename + ".txt", "w")
    f.write("chrom,position,alleles,risk,expected_coef,expected_se,"
            "observed_coef,observed_se,observed_maf,n\n")

    for variant, statistic in variant_to_expected.items():
        if not statistic.valid:
            logger.warning(statistic.message)

        else:
            xs.append(statistic.e_coef)
            ys.append(statistic.o_coef)

            if not no_error_bars:
                xs_error.append(statistic.e_error)
                ys_error.append(statistic.o_error)

            # File
            line = [str(variant.chrom), str(variant.pos),
                    "/".join(variant.alleles_set), statistic.e_risk,
                    str(statistic.e_coef), str(statistic.e_error),
                    str(statistic.o_coef), str(statistic.o_error),
                    str(statistic.o_maf),
                    str(statistic.o_nobs)]

            line = ",".join(line)
            f.write(line + "\n")

    f.close()

    # Plot
    if not no_error_bars:
        plt.errorbar(xs, ys, xerr=xs_error, yerr=ys_error,
                     fmt='.', color='black', markersize=3, capsize=2,
                     markeredgewidth=0.5, elinewidth=0.5, ecolor='0.7',
                     label="coefficients")
    else:
        plt.plot(xs, ys, '.', color='black', markersize=3,
                 label="coefficients")

    plt.xlabel('Expected coefficients')
    plt.ylabel('Observed coefficients')

    # Correlation
    slope, intercept, rval, pval, std_err = scipy.stats.linregress(xs, ys)

    xmin = np.min(xs)
    xmax = np.max(xs)

    x = np.linspace(xmin, xmax, 2000)

    plt.plot(x, slope * x + intercept,
             label=("observed_coef = {:.2f} expected_coef + {:.2f} "
                    "($R^2={:.2f}$)".format(slope, intercept, rval ** 2)),
             linewidth=0.5)

    plt.plot(x, x, label="observed_coef = expected_coef", linestyle="--",
             linewidth=0.5, color='blue')

    plt.legend()

    if svg_format:
        plt.savefig(out_filename + ".svg")
    else:
        plt.savefig(out_filename + ".png", dpi=300)
