"""
Compute the GRS from genotypes and a GRS file.
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

import numpy as np
import pandas as pd
import geneparse
import geneparse.config
from geneparse.core import complement_alleles

from ..utils import (parse_grs_file, parse_kwargs, clopper_pearson_interval,
                     find_tag)


DEBUG = False


logger = logging.getLogger(__name__)


class VariantNotInReference(KeyError):
    pass


class VariantDupOrMulti(Exception):
    pass


class CouldNotFindTag(Exception):
    pass


class ScoreInfo(object):
    __slots__ = ("effect", "reference", "risk")

    def __init__(self, effect, reference, risk):
        self.effect = effect
        self.reference = reference
        self.risk = risk


def _weight_unambiguous(g, info, quality_weight):
    """Compute the GRS constribution for a variant of unambiguous strand."""
    # Weight by quality if needed.
    if quality_weight and isinstance(g.variant, geneparse.ImputedVariant):
        info.effect *= g.variant.quality

    # "Impute" missing genotypes by setting them to the sample mean.
    g.genotypes[np.isnan(g.genotypes)] = np.nanmean(g.genotypes)

    if g.coded == info.risk and g.reference == info.reference:
        # No need to flip.
        pass

    elif g.coded == info.reference and g.reference == info.risk:
        g.flip()

    else:
        raise RuntimeError(
            "Unexpected allele mismatch during GRS computation: "
            "{} vs {}.".format(
                {g.coded, g.reference}, {info.risk, info.reference}
            )
        )

    # Always use the allele with a positive effect when adding to the
    # score.
    if info.effect < 0:
        cur = (2 - g.genotypes) * -info.effect
    else:
        cur = g.genotypes * info.effect

    return cur


def _id_strand_by_frequency(g, reference):
    """Identifies strand based on allele frequency.

    This function validates the strand from observed genotypes and a reference
    panel. It compares the allele frequencies based on a Clopper Pearson
    confidence interval from the refrence frequencies. If the computed interval
    does not include 0.5 and includes the observed frequency, we consider the
    strand to be validated and False is returned (need to flip = False).

    If the complementary frequency falls in the confidence interval, the strand
    is considered to be validated but the alleles need to be flipped (need to
    flip = True is returned).

    If the strand can't be validated based on the frequency, None is returned.

    The VariantNotInReference exception is raised if the variant can't be found
    in the reference panel.

    The VariantDupOrMulti exception is raised if more than one reference
    genotypes are returned by geneparse when queried with the observed
    Variant instance.

    Args:
        g (Genotype): The observed genotypes
        reference (geneparse.core.GenotypesReader): An initialized geneparse
            reader to reference panel genotypes.

    Returns:
        bool or None: If the alleles could be identified based on the binomial
            confidence interval, the return value will be True if we need to
            flip the alleles and False otherwise. If the alleles could not
            be identified, the return value will be None

    """

    # Get the variant
    ref_g = reference.get_variant_genotypes(g.variant)

    if len(ref_g) == 0:
        raise VariantNotInReference(g)

    elif len(ref_g) != 1:
        raise VariantDupOrMulti(g)

    assert len(ref_g) == 1
    ref_g = ref_g[0]

    # Check that the frequencies are not too close to 0.5.
    if ref_g.maf() > 0.4 or g.maf() > 0.4:
        if DEBUG:
            logger.debug(
                "{} MAF too close to 50% (data_maf={:.3f}, ref_maf={:.3f})"
                "".format(g.variant, g.maf(), ref_g.maf())
            )
        return

    # Compare the alleles.
    if g.coded == ref_g.coded:
        pass

    else:
        g.flip()
        if g.coded != ref_g.coded:
            raise RuntimeError(
                "Unexpected allele mismatch during GRS computation."
            )

    assert g.coded == ref_g.coded

    # Compute the confidence interval over the reference frequency.
    low, high = clopper_pearson_interval(
        np.nansum(ref_g.genotypes),
        2 * np.sum(~np.isnan(ref_g.genotypes))
    )

    g_coded_freq = g.coded_freq()

    if DEBUG:
        logger.debug(
            "{} reference f({})={:.3f} ({:.3f}, {:.3f}); "
            "data f({})={:.3f}."
            "".format(
                g.variant,
                ref_g.coded, ref_g.coded_freq(), low, high,
                g.coded, g_coded_freq
            )
        )

    if low <= g_coded_freq <= high:
        # The coded alleles match.
        # First, we make a sanity check that the other allele combination
        # would not match.
        if low <= 1 - g_coded_freq <= high:
            return

        return False

    elif low <= 1 - g_coded_freq <= high:
        # We need to flip.
        return True

    else:
        # There is a frequency mismatch.
        return


def _replace_by_tag(g, info, reference, reader, r2_threshold=0.6):

    tag = find_tag(reference, g.variant, extract_reader=reader)
    if tag is None:
        raise CouldNotFindTag()

    # Unpack the tag information.
    # tag will be a Genotypes instance.
    g, tag, r = tag

    # Check if R2 is high enough.
    r2 = r ** 2
    if r2 < r2_threshold:
        raise CouldNotFindTag(
            "Best tag had R2={:.2f} < {:.2f}"
            "".format(r2, r2_threshold)
        )

    # Identify the allele tagging the effect allele.
    # Note: ScoreInfo(effect, reference, risk)
    tag_effect = info.effect * r2
    if info.risk == g.coded:
        # If r is positive, then tag coded == effect
        if r > 0:
            tag_info = ScoreInfo(tag_effect, tag.reference, tag.coded)
        else:
            tag_info = ScoreInfo(tag_effect, tag.coded, tag.reference)

    elif info.risk == g.reference:
        # If r is negative, then tag coded == effect
        if r > 0:
            tag_info = ScoreInfo(tag_effect, tag.coded, tag.reference)
        else:
            tag_info = ScoreInfo(tag_effect, tag.reference, tag.coded)

    else:
        raise RuntimeError(
            "Unexpected allele mismatch during GRS computation."
        )

    # Find the tag in the genotypes file.
    g = reader.get_variant_genotypes(tag.variant)
    assert len(g) == 1, "This should be guaranteed by find_tag."
    g = g[0]

    return g, tag_info, r2


def _weight_ambiguous(g, info, quality_weight, reference, reader):
    need_strand_flip = _id_strand_by_frequency(g, reference)
    if need_strand_flip is True:
        logger.info(
            "STRAND FLIPPED: {} {} (based on frequency)".format(
                g.variant.name, g.variant
            )
        )

        # We flip the allele labels.
        g.reference, g.coded = g.coded, g.reference
        return _weight_unambiguous(g, info, quality_weight)

    elif need_strand_flip is False:
        logger.info(
            "STRAND VALIDATED: {} {} (based on frequency)".format(
                g.variant.name, g.variant
            )
        )

        # We are on the right strand.
        return _weight_unambiguous(g, info, quality_weight)

    else:
        logger.debug(
            "STRAND NOT VALIDATED (frequency mismatch). Trying to find a tag."
        )

    # We need to find a tag instead.
    tag, tag_info, r2 = _replace_by_tag(g, info, reference, reader)
    logger.info(
        "TAG:{}: {} substitutes SRC:{}: {} (reference R2={:.2f})"
        "".format(
            tag.variant.name, tag.variant,
            g.variant.name, g.variant,
            r2
        )
    )

    return _weight_unambiguous(tag, tag_info, quality_weight)


def compute_grs(reader, samples, genotypes_and_info, quality_weight=True,
                skip_strand_check=False, exclude_strand_ambiguous=False,
                reference=None):
    quality_weight_warned = False

    n_variants_used = 0
    grs = np.zeros(len(samples))

    for g, info in genotypes_and_info:
        # Warn that we will weight the variants by quality.
        if isinstance(g.variant, geneparse.ImputedVariant) and quality_weight:
            if not quality_weight_warned:
                quality_weight_warned = True
                logger.info("WEIGHTING score by variant quality.")

        # Dispatch to the correct weighting function wrt strand ambiguity.
        if not g.variant.alleles_ambiguous():
            grs += _weight_unambiguous(g, info, quality_weight)

        elif skip_strand_check:
            grs += _weight_unambiguous(g, info, quality_weight)

        elif exclude_strand_ambiguous:
            logger.info("EXCLUDING ambiguous variant {} {}."
                        "".format(g.variant.name, g.variant))
            continue

        else:
            try:
                cur = _weight_ambiguous(g, info, quality_weight, reference,
                                        reader)
                grs += cur

            except VariantNotInReference:
                logger.warning(
                    "Could not identify strand for {} {} (variant not in "
                    "reference).".format(g.variant.name, g.variant)
                )
                continue

            except VariantDupOrMulti:
                logger.warning(
                    "Could not identify strand for {} {} (variant is a "
                    "multiallelic or has duplicates)."
                    "".format(g.variant.name, g.variant)
                )
                continue

            except CouldNotFindTag:
                logger.warning(
                    "EXCLUDING {} {} because impossible to identify strand or "
                    "find a suitable tag."
                    "".format(g.variant.name, g.variant)
                )
                continue

        n_variants_used += 1

    logger.info("Computed the GRS using {} variants.".format(n_variants_used))
    return pd.DataFrame(grs, index=samples, columns=["grs"])


def main():
    args = parse_args()

    if args["reference"]:
        reference = geneparse.parsers["plink"](args["reference"])
    else:
        reference = None

    # Parse the GRS.
    grs = parse_grs_file(args["grs"])

    # Extract genotypes.
    if args["genotypes_format"] not in geneparse.parsers.keys():
        raise ValueError(
            "Unknown reference format '{}'. Must be a genetest compatible "
            "format ({})."
            "".format(args["genotypes_format"], list(geneparse.parsers.keys()))
        )

    geneparse.config.LOG_NOT_FOUND = False
    reader = geneparse.parsers[args["genotypes_format"]]

    kwargs = parse_kwargs(args["genotypes_kwargs"])
    kwargs = kwargs if kwargs is not None else {}
    reader = reader(args["genotypes"], **kwargs)

    logger.info("Extracting genotypes for variants in the GRS file.")
    # List of tuples of (genotype, info) instances
    genotypes_and_info = []
    for name, row in grs.iterrows():
        v = geneparse.Variant(
            name, row.chrom, row.pos, [row.reference, row.risk]
        )

        info = ScoreInfo(
            row.effect, row.reference, row.risk
        )

        # Get the genotype.
        g = reader.get_variant_genotypes(v)

        if len(g) == 0:
            # Maybe the other strand is on the chip.
            v.complement_alleles()
            info.reference = complement_alleles(info.reference)
            info.risk = complement_alleles(info.risk)
            g = reader.get_variant_genotypes(v)

            if len(g) == 0:
                logger.warning("Excluding {} {} (no available genotypes)."
                               "".format(name, v))
                continue

            else:
                logger.info(
                    "Found variant {} {} after complementation (on the other "
                    "strand).".format(name, v)
                )

        if len(g) == 1:
            genotypes_and_info.append((g[0], info))

        else:
            logger.warning(
                "Excluding {} {} (duplicate variant or ambiguous multiallelic)"
                "".format(name, v)
            )

    computed_grs = compute_grs(
        reader,
        reader.get_samples(),
        genotypes_and_info,
        quality_weight=not args["ignore_genotype_quality"],
        skip_strand_check=args["skip_strand_check"],
        exclude_strand_ambiguous=args["exclude_strand_ambiguous"],
        reference=reference
    )

    logger.info("WRITING file containing the GRS: '{}'".format(args["out"]))
    computed_grs.to_csv(args["out"], header=True, index_label="sample")


def startup_log(args):
    logger.info("grs-compute called with the following options:")
    logger.info("\tGenotype data file: {}".format(args["genotypes"]))
    logger.info("\tGenotype data file format: {}".format(
        args["genotypes_format"])
    )
    if args["genotypes_kwargs"] is not None:
        logger.info(
            "\tKeyword arguments passed to the genotypes reader: {}"
            "".format(args["genotypes_kwargs"])
        )
    logger.info("\tGRS file (selected variants): {}".format(args["grs"]))
    logger.info("\tOutput filename: {}".format(args["out"]))
    logger.info(
        "\tVariants {} be weighted by the variant quality (if available)"
        "".format("will not" if args["ignore_genotype_quality"] else "will")
    )
    if args["skip_strand_check"]:
        logger.info("\tStrand checks will be SKIPPED and the variants will "
                    "be assumed to be on the same strand as the GRS file.")
    elif args["exclude_strand_ambiguous"]:
        logger.info("\tVariants with ambiguous strands will be automatically "
                    "excluded.")
    else:
        assert args["reference"] is not None
        logger.info("\tReference panel used for strand checks: '{}'"
                    "".format(args["reference"]))


def parse_args():
    global DEBUG

    parser = argparse.ArgumentParser(description="Compute the risk score.")

    parser.add_argument(
        "--genotypes",
        help="File containing genotype data.",
        type=str
    )

    parser.add_argument(
        "--genotypes-format",
        type=str,
        help=("File format of the genotypes in the reference (default: "
              "%(default)s)."),
        default="plink"
    )

    parser.add_argument(
        "--genotypes-kwargs",
        type=str,
        help=("Keyword arguments to pass to the genotypes container. "
              "A string of the following format is expected: "
              "'key1=value1,key2=value2,..."
              "It is also possible to prefix the values by 'int:' or 'float:' "
              "to cast the them before passing them to the constructor.")
    )

    parser.add_argument(
        "--grs",
        help=("File describing the variants in the GRS. "
              "The expected columns are 'name', 'beta', 'reference', 'risk'."),
        type=str,
        required=True
    )

    parser.add_argument(
        "--out",
        help="Filename for the computed GRS (default: %(default)s).",
        default="computed_grs.csv"
    )

    parser.add_argument(
        "--ignore-genotype-quality",
        help=("For imputed variants, if this flag is set, the variants "
              "will not be weighted with respect to their quality score. "
              "By default, the weights are used if available."),
        action="store_true"
    )

    parser.add_argument(
        "--skip-strand-check",
        help=("Skips all the strand validation checks and assume that the "
              "strand in the genotype and GRS files are the same. "
              "This option should only be used if the variants were manually "
              "curated. "
              "This option is also assumed if no --reference is provided "
              "(with a warning)."),
        action="store_true"
    )

    parser.add_argument(
        "--exclude-strand-ambiguous",
        help="Exclude all strand ambiguous variants.",
        action="store_true"
    )

    parser.add_argument(
        "--reference",
        default=None,
        type=str,
        help=("Plink prefix for the reference genotypes. This is used to find "
              "tags when needed or to compute the MAF.")
    )

    parser.add_argument(
        "--debug",
        action="store_true"
    )

    args = vars(parser.parse_args())

    if args["debug"]:
        DEBUG = True
        logger.setLevel(logging.DEBUG)

    if args["exclude_strand_ambiguous"] and args["skip_strand_check"]:
        raise ValueError(
            "Either --exclude-strand-ambiguous to ignore variants whose "
            "is ambiguous OR --skip-strand-check to assume that the strands "
            "are the same between the GRS file and the genotypes file."
        )

    no_exception = (
        args["exclude_strand_ambiguous"] is False and
        args["skip_strand_check"] is False
    )
    if args["reference"] is None and no_exception:
        raise ValueError(
            "Can't execute strand checks because no reference panel was provided. "
            "To run the analysis anyway, either provide a '--reference' "
            "option, use '--skip-strand-check' or use "
            "'--exclude-strand-ambiguous'."
        )

    startup_log(args)

    return args
