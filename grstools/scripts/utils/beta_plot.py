import logging

import geneparse
import genetest.modelspec as spec
from genetest.phenotypes import TextPhenotypes
from genetest.analysis import execute
from genetest.statistics import model_map
from genetest.subscribers import Subscriber
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


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
        self.variant_to_remove = set()

    def handle(self, results):
        v = geneparse.Variant("None",
                              results["SNPs"]["chrom"],
                              results["SNPs"]["pos"],
                              [results["SNPs"]["major"],
                               results["SNPs"]["minor"]])

        if results["SNPs"]["maf"] < 0.01:
            logger.warning("Ignoring {} because its maf ({}) is "
                           "less than 1%".format(v, results["SNPs"]["maf"]))

            self.variant_to_remove.add(v)

            return

        # Same reference and risk alleles for expected and observed
        if self.variant_to_expected[v].e_risk == results["SNPs"]["minor"]:
            self.variant_to_expected[v].o_risk = results["SNPs"]["minor"]
            self.variant_to_expected[v].o_coef = results["SNPs"]["coef"]

        else:
            self.variant_to_expected[v].o_risk = results["SNPs"]["major"]
            self.variant_to_expected[v].o_coef = -results["SNPs"]["coef"]

        self.variant_to_expected[v].o_error = results["SNPs"]["std_err"]
        self.variant_to_expected[v].o_maf = results["SNPs"]["maf"]
        self.variant_to_expected[v].o_nobs = results["MODEL"]["nobs"]


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
        test=test_specification
    )

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

    # Remove variants with maf less than 1%
    for v in custom_sub.variant_to_remove:
        del variant_to_expected[v]

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

    plt.savefig(args.out + ".png", dpi=500)
