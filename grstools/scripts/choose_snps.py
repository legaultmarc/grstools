"""
Choose SNPs from GWAS summary statistics.

Method 1:
    Sort by significance.
    Choose top SNP.
    Remove all SNPs in LD.
    Loop until p-value threshold or n variants is reached.

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


import argparse
import logging
import pickle
import time
import os
import csv
import collections

import geneparse
import geneparse.utils
from geneparse.exceptions import InvalidChromosome

from ..utils import parse_grs_file, InMemoryGenotypeExtractor
from ..version import grstools_version


logger = logging.getLogger(__name__)


class ParameterObject(object):
    """Abstract class of argument validators."""
    @staticmethod
    def valid(o):
        raise NotImplementedError()


class Filename(ParameterObject):
    @staticmethod
    def valid(o):
        """Validate that the object can be opened for reading."""
        try:
            with open(o, "r"):
                pass
        except:
            return False
        return True


class Region(ParameterObject):
    @classmethod
    def valid(cls, o):
        """Validate that a str is of the form chrXX:START-END."""
        try:
            cls._parse_region(o)
        except ValueError:
            return False
        return True

    @staticmethod
    def _parse_region(s):
        message = "Expected format for region is: 'chr1:12345-22345'."
        if not hasattr(s, "startswith"):
            raise ValueError(message)

        if not s.startswith("chr"):
            raise ValueError(message)

        s = s[3:]
        try:
            chrom, tail = s.split(":")
            start, end = [int(i) for i in tail.split("-")]
        except:
            raise ValueError(message)

        # Flip start and end position if needed.
        if start > end:
            start, end = end, start

        return chrom, start, end


class Some(ParameterObject):
    def __new__(cls, t):
        # Valid if:
        #  - Is none
        #  - Is instance of t
        #  - t.valid is true

        def valid(o):
            if o is None:
                return True

            if isinstance(o, t):
                return True

            if hasattr(t, "valid") and t.valid(o):
                return True

            return False

        return type(
            "Some",
            (ParameterObject, ),
            {"valid": valid}
        )


class SNPSelectionLog(object):
    """Class to keep a log of selected SNPs for a GRS."""
    # Known parameters and their types.
    KNOWN_PARAMETERS = {
        "DEBUG": bool,
        "SUMMARY_FILENAME": Filename,
        "REFERENCE_FILENAME": str,
        "MAF_THRESHOLD": float,
        "TARGET_N": Some(int),
        "LD_WINDOW_SIZE": int,
        "LD_CLUMP_THRESHOLD": float,
        "P_THRESHOLD": float,
        "EXCLUDE_AMBIGUOUS_ALLELES": bool,
        "EXCLUDE_NO_REFERENCE": bool,
        "REGION_INCLUDED": Some(Region),
        "REGION_EXCLUDED": Some(Region),
        "OUTPUT_PREFIX": str,
    }

    EXCLUSION_REASONS = {
        "MAF_FILTER",
        "INVALID_CHROMOSOME",
        "AMBIGUOUS_ALLELES_FILTER",
        "REGION_INCLUSION_FILTER",
        "REGION_EXCLUSION_FILTER",
        "NOT_IN_REFERENCE_PANEL",
        "DUP_OR_MULTIALLELIC_IN_REFERENCE_PANEL",
        "LD_CLUMPED"
    }

    def __init__(self, output_prefix):
        self.parameters = {"OUTPUT_PREFIX": output_prefix}
        self.available_variants = None
        self.special = []
        self.included = []
        self.excluded = []

    def init_logger(self):
        """Sets the log level and binds to a logger instance.

        This is called automatically as a sort of startup hook after argument
        parsing.

        """
        self.logger = logger

        if self.parameters.get("DEBUG", False):
            self.logger.setLevel(logging.DEBUG)

        self.display_configuration()

    def log_selection_trace(self):
        """Gets called at the end of the selection process.

        This function is responsible for writing the exclusions to disk.

        """
        self.logger.info("Writing selection logs (exclusions and special "
                         "warnings).")

        excluded_filename = self.parameters["OUTPUT_PREFIX"] + ".excluded.log"
        special_incl_filename = (
            self.parameters["OUTPUT_PREFIX"] + ".special.log"
        )

        variant_sets = [
            (excluded_filename, self.excluded),
            (special_incl_filename, self.special)
        ]

        for filename, variant_set in variant_sets:
            with open(filename, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "chrom", "pos", "alleles", "reason",
                                 "details"])

                for variant, reason, details in variant_set:
                    alleles = "/".join(variant.alleles)
                    writer.writerow([
                        variant.name, variant.chrom, variant.pos,
                        alleles, reason, details
                    ])

    def close(self):
        pass

    def display_configuration(self):
        """Displays all recorded parameter values.

        This is called by init_logger.

        """
        self.logger.info("grstools v{}".format(grstools_version))

        self.logger.info("Starting variant selection with the following "
                         "parameters:")
        self.logger.info(
            "\tUsing summary statistics from: '{}'"
            "".format(self.parameters["SUMMARY_FILENAME"])
        )

        self.logger.info(
            "\tReference panel for MAF and LD computation are from: '{}'"
            "".format(self.parameters["REFERENCE_FILENAME"])
        )

        self.logger.info(
            "\tP-value <= {:g}"
            "".format(self.parameters["P_THRESHOLD"])
        )

        self.logger.info(
            "\tMAF >= {:.4f}"
            "".format(self.parameters["MAF_THRESHOLD"])
        )

        if self.parameters.get("TARGET_N"):
            self.logger.info(
                "\tSelecting up to {} variants"
                "".format(self.parameters["TARGET_N"])
            )

        self.logger.info(
            "\tClumping variants with LD >= {:.2f}"
            "".format(self.parameters["LD_CLUMP_THRESHOLD"])
        )

        self.logger.info(
            "\t{} variants with ambiguous alleles (A/T or G/C)"
            "".format(
                "EXCLUDING" if self.parameters["EXCLUDE_AMBIGUOUS_ALLELES"]
                else "INCLUDING"
            )
        )

        self.logger.info(
            "\t{} variants that are absent from the reference genotypes"
            "".format(
                "EXCLUDING" if self.parameters["EXCLUDE_NO_REFERENCE"]
                else "INCLUDING"
            )
        )

        if self.parameters.get("REGION_INCLUDED"):
            self.logger.info(
                "\tINCLUDING variants in region '{}'"
                "".format(self.parameters["REGION_INCLUDED"])
            )

        if self.parameters.get("REGION_EXCLUDED"):
            self.logger.info(
                "\tEXCLUDING variants in region '{}'"
                "".format(self.parameters["REGION_EXCLUDED"])
            )

        self.logger.info("\tOutput prefix: '{}'"
                         "".format(self.parameters["OUTPUT_PREFIX"]))

    # Changes in state that are to be recorded by the logger.
    def record_parameter(self, key, value):
        """Record the value of parameter of the algorithm."""
        if key not in self.KNOWN_PARAMETERS:
            raise ValueError("Unknown parameter '{}'.".format(key))

        t = self.KNOWN_PARAMETERS[key]
        if isinstance(value, t):
            # Provided with an instance of the right type.
            pass

        elif hasattr(t, "valid") and t.valid(value):
            # Value was validated by ParameterObject.
            pass

        else:
            raise ValueError(
                "Invalid value '{}' for parameter '{}'."
                "".format(value, key)
            )

        self.parameters[key] = value

    def record_included_special(self, variant, reason, details=None):
        """Record included variants that need special care or attention.

        For example, ambiguous variants that have a frequency close to 0.5
        or variants that are absent from the reference panel.

        """
        self.special.append((variant, reason, details))

    def record_included(self, variant):
        """Record that a variant has been included in the GRS."""
        self.included.append(variant)

    def record_excluded(self, variant, reason, details=None):
        """Record that a variant has been excluded (and why)."""
        if reason not in self.EXCLUSION_REASONS:
            raise ValueError(
                "Unknown reason for exclusion: '{}'"
                "".format(reason)
            )
        self.excluded.append((variant, reason, details))

    def record_ld_block(self, variant, other_loci, r2):
        """Record the LD between variants."""
        if len(other_loci) == 0:
            self.logger.debug("\tVARIANT {} ALONE".format(variant))
            return

        other_loci = [g.variant for g in other_loci]

        start = variant.pos - self.parameters["LD_WINDOW_SIZE"] // 2
        end = variant.pos + self.parameters["LD_WINDOW_SIZE"] // 2

        self.logger.debug(
            "\tLD_REGION {} to {} ({}: {}-{}) [{} candidates]"
            "".format(
                other_loci[0],
                other_loci[-1],
                variant.chrom,
                start, end, len(other_loci))
        )

        blocks_directory = self.parameters["OUTPUT_PREFIX"] + ".ld_blocks"
        if not os.path.isdir(blocks_directory):
            os.mkdir(blocks_directory)

        # Serialize LD blocks to disk.
        filename = os.path.join(
            blocks_directory,
            "variant_{}_{}:{}-{}.blocks.pkl".format(
                variant.name, variant.chrom, start, end
            )
        )

        with open(filename, "wb") as f:
            pickle.dump({
                "cur": variant,
                "other_loci": other_loci,
                "r2": r2
            }, f)

    def get_excluded_variants(self):
        return {i[0] for i in self.excluded}


class Row(object):
    __slots__ = ("name", "chrom", "pos", "reference", "risk", "p_value",
                 "effect", "maf")

    def __init__(self, name, chrom, pos, reference, risk, p_value, effect,
                 maf=None):
        """The row of a GRS file."""
        self.name = name
        self.chrom = chrom
        self.pos = pos
        self.reference = reference
        self.risk = risk
        self.p_value = p_value
        self.effect = effect
        self.maf = maf

    def write_header(self, f):
        f.write("name,chrom,pos,reference,risk,p-value,effect")
        if self.maf is not None:
            f.write(",maf")
        f.write("\n")

    @property
    def _fields(self):
        fields = [
            self.name, self.chrom, self.pos, self.reference, self.risk,
            self.p_value, self.effect
        ]

        if self.maf is not None:
            fields.append(self.maf)

        return fields

    def write(self, f, sep=","):
        for i, field in enumerate(self._fields):
            if i != 0:
                f.write(",")

            if type(field) is float:
                f.write("{:.9g}".format(field))
            else:
                f.write(str(field))

        f.write("\n")


def read_summary_statistics(filename, p_threshold, log, sep=",",
                            exclude_ambiguous=False, region=None,
                            exclude_region=None):
    """Read summary statistics file.

    Args:
        filename (str): Summary statistics (.grs) file name.
        p_threshold (float): Minimum p-value for inclusion.
        sep (str): File column delimiter.
        exclude_ambiguous (bool): Flag to exclude ambiguous (A/T or G/C)
            variants.
        region (str): Genomic region of the form chr3:12345-12355. If a region
            is provided, only variants in the region will be KEPT.
        exclude_region (str): Genomic region to exclude (see above for
            details).

    Returns:
        collections.OrderedDict: The OrderedDict maps Variant instances to
            their summary statistics (effect size, p-value, etc.) represented
            as Row instances.

    """
    if region is not None:
        region = Region._parse_region(region)

    if exclude_region is not None:
        exclude_region = Region._parse_region(exclude_region)

    # Variant to stats orderedict (but constructed as a list).
    summary = []

    df = parse_grs_file(filename, p_threshold=p_threshold, sep=sep)
    df.sort_values("p-value", inplace=True)

    # Method to see if a variant is in a region.
    def _in_region(variant, chrom, start, end):
        return (
            variant.chrom == chrom and
            start <= variant.pos <= end
        )

    # For now, this is not a limiting step, but it might be nice to parallelize
    # this eventually.
    for idx, info in df.iterrows():
        if info["p-value"] > p_threshold:
            break

        try:
            variant = geneparse.Variant(info["name"], info.chrom, info.pos,
                                        [info.reference, info.risk])
        except InvalidChromosome:
            bad_v = geneparse.Variant(
                info["name"], geneparse.Chromosome(info.chrom), info.pos,
                [info.reference, info.risk]
            )
            log.record_excluded(bad_v, "INVALID_CHROMOSOME", info.chrom)
            continue

        # Region based inclusion/exclusion
        if region is not None:
            if not _in_region(variant, *region):
                log.record_excluded(
                    variant, "REGION_INCLUSION_FILTER",
                    "{} not in {}".format(variant, region)
                )
                continue

        if exclude_region is not None:
            if _in_region(variant, *exclude_region):
                log.record_excluded(
                    variant, "REGION_EXCLUSION_FILTER",
                    "{} in {}".format(variant, exclude_region)
                )
                continue

        ambiguous = variant.alleles_ambiguous()
        if ambiguous and exclude_ambiguous:
            log.record_excluded(variant, "AMBIGUOUS_ALLELES_FILTER")
            continue

        if "maf" in info.index:
            if info.maf <= log.parameters["MAF_THRESHOLD"]:
                log.record_excluded(
                    variant,
                    "MAF_FILTER",
                    "MAF recorded as {:g} in summary statistics file"
                    "".format()
                )
                continue

        row_args = [info["name"], info.chrom, info.pos, info.reference,
                    info.risk, info["p-value"], info.effect]

        if "maf" in info.index:
            row_args.append(info.maf)

        row = Row(*row_args)
        summary.append((variant, row))

    # Convert the summary statistics to an ordereddict of loci to stats.
    summary = collections.OrderedDict(
        sorted(summary, key=lambda x: x[1].p_value)
    )

    return summary


def greedy_pick_clump(summary, genotypes, log):
    """Greedy algorithm to select SNPs for inclusion in the GRS.

    Args:
        summary (Dict[Variant, Row]): Dict representation of the summary
            statistics file containing Variants as keys and their information
            as values.
        genotypes (geneparse.core.GenotypesReader): Genotypes from a reference
            panel for LD computation.
        log (SNPSelectionLog): A class to manage state of the selection
            process.
        target_n (int): Number of variants to stop the selection routine. This
            can be used if only the top N SNPs should be used to define the
            GRS.

    Returns:
        List[Row]: Variants selected by the algorithm.

    """
    log.logger.info("Starting the greedy variant selection.")

    out = []
    while len(summary) > 0:

        # One of the stop conditions is target n, which we check.
        target_n = log.parameters.get("TARGET_N")
        if target_n is not None and len(out) >= target_n:
            log.logger.info("Target number of variants reached.")
            break

        # Get the next best variant.
        cur, info = summary.popitem(last=False)

        # Check if current variant is a suitable candidate.
        g = genotypes.get_variant_genotypes(cur)

        if variant_is_good_to_keep(cur, g, log):
            log.record_included(cur)
            out.append(info)

            # If the variant is in the reference, we do LD pruning.
            if len(g) == 1:
                summary = ld_prune(summary, g[0], genotypes, log)

        # Otherwise, just go to the next one (exclusion will be noted by the
        # predicate function).

    return out


def variant_is_good_to_keep(cur, g, log):
    """Check that the currently selected variant is good to keep.

    When this is called, it's after parsing the summary statistics and when
    looking up variants in the reference panel.

    This means that the filters being applied are:

    - Reference panel MAF filtering
    - Filters based on the availability in reference panels
    - Filters based on multiallelic / duplicates in reference panel

    """
    # Variant not in reference panel
    if len(g) == 0:
        if log.parameters["EXCLUDE_NO_REFERENCE"]:
            log.record_excluded(cur, "NOT_IN_REFERENCE_PANEL")
            return False

        else:
            log.record_included_special(
                cur,
                "NOT_IN_REFERENCE",
                "Variant {} was absent from the reference panel but was still "
                "included in the GRS. It is important to validate that it is "
                "not correlated with other included variants."
                "".format(cur)
            )
            return True

    elif len(g) == 1:
        # Variant was uniquely found in the reference panel, we can check the
        # MAF.
        maf = g[0].maf()
        if maf <= log.parameters["MAF_THRESHOLD"]:
            log.record_excluded(
                cur,
                "MAF_FILTER",
                "MAF from reference panel is {:g}".format(maf)
            )
            return False

    # Variant is duplicate or multiallelic
    elif len(g) > 1:
        log.record_excluded(cur, "DUP_OR_MULTIALLELIC_IN_REFERENCE_PANEL")
        return False

    return True


def ld_prune(summary, g, genotypes, log):
    """Return a list of variant with all variants correlated to cur removed."""
    v = g.variant
    left = v.pos - log.parameters["LD_WINDOW_SIZE"] // 2
    right = v.pos + log.parameters["LD_WINDOW_SIZE"] // 2

    others = list(genotypes.get_variants_in_region(v.chrom, left, right))

    # Remove the variants in reference but not in summary.
    others = [
        other_g for other_g in others if other_g.variant in summary.keys()
    ]

    if len(others) < 1:
        # No need to prune, no variants in LD.
        return summary

    # r2 is a series with index the variant name in the reference file.
    r2 = geneparse.utils.compute_ld(g, others, r2=True)

    # Remove all the variants from the summary statistics if correlated.
    ld_threshold = log.parameters["LD_CLUMP_THRESHOLD"]

    # Record the LD matrix.
    log.record_ld_block(v, others, r2.values)

    for g, ld in zip(others, r2):
        if ld >= ld_threshold:
            del summary[g.variant]

            log.record_excluded(
                g.variant,
                "LD_CLUMPED",
                "LD with {} {} is {:g}".format(
                    v.name if v.name else "",
                    v, ld
                )
            )

    return summary


def write_selection(grs, log):
    """Write the selected SNPs to disk."""
    if len(grs) == 0:
        log.logger.warning(
            "No variant satisfied the provided thresholds (could not generate "
            "a GRS)."
        )
        return

    log.logger.info(
        "Writing the file containing the selected {} variants."
        "".format(len(grs))
    )

    output_filename = log.parameters["OUTPUT_PREFIX"] + ".grs"
    with open(output_filename, "w") as f:
        grs[0].write_header(f)
        for row in grs:
            row.write(f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--p-threshold",
        help="P-value threshold for inclusion in the GRS (default: 5e-8).",
        default=5e-8,
        type=float
    )

    parser.add_argument(
        "--target-n",
        help="Target number of variants to include in the GRS.",
        default=None,
        type=int
    )

    parser.add_argument(
        "--maf-threshold",
        help="Minimum MAF to allow inclusion in the GRS (default %(default)s).",
        default=0.05,
        type=float
    )

    parser.add_argument(
        "--ld-threshold",
        help=("LD threshold for the clumping step. All variants in LD with "
              "variants in the GRS are excluded iteratively (default "
              "%(default)s)."),
        default=0.15,
        type=float
    )

    parser.add_argument(
        "--ld-window-size",
        help=("Size of the LD window used to find correlated variants. "
              "Making this window smaller will make the execution faster but "
              "increases the chance of missing correlated variants "
              "(default 1Mb)."),
        default=int(1e6),
        type=int
    )

    parser.add_argument(
        "--region",
        help=("Only consider variants located WITHIN a genomic region. "
              "The expected format is 'chrCHR:START-END'. For example: "
              "'chr1:12345-22345'."),
        default=None,
        type=str
    )

    parser.add_argument(
        "--exclude-region",
        help=("Only consider variants located OUTSIDE a genomic region. "
              "The expected format is 'chrCHR:START-END'. For example: "
              "'chr1:12345-22345'."),
        default=None,
        type=str
    )

    parser.add_argument(
        "--exclude-ambiguous-alleles",
        help="Exclude variants with ambiguous alleles (e.g. G/C or A/T)",
        action="store_true"
    )

    parser.add_argument(
        "--exclude-no-reference",
        help="Exclude variants with no genotypes in the reference panel.",
        action="store_true"
    )

    # Files
    parser.add_argument(
        "--summary",
        help=("Path to the summary statistics files. Required columns are "
              "'name', 'chrom', 'pos', 'p-value', 'effect', 'reference' and "
              "'risk'."),
        required=True
    )

    parser.add_argument(
        "--reference",
        help=("Path the binary plink file containing reference genotypes. "
              "These genotypes will be used for LD clumping."),
        required=True
    )

    parser.add_argument(
        "--output", "-o",
        help="Output prefix (default: %(default)s).",
        default="grstools_selection"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    return parser.parse_args()


def main():
    SCRIPT_START_TIME = time.time()

    global debug

    args = parse_args()

    # Setting the output prefix.
    log = SNPSelectionLog(args.output)

    if args.debug:
        debug = True
        log.record_parameter("DEBUG", True)
        logger.setLevel(logging.DEBUG)

    summary_filename = args.summary
    log.record_parameter("SUMMARY_FILENAME", summary_filename)

    reference_filename = args.reference
    log.record_parameter("REFERENCE_FILENAME", reference_filename)

    # Parameters
    p_threshold = args.p_threshold
    log.record_parameter("P_THRESHOLD", p_threshold)

    target_n = args.target_n
    log.record_parameter("TARGET_N", target_n)

    maf_threshold = args.maf_threshold
    log.record_parameter("MAF_THRESHOLD", maf_threshold)

    ld_threshold = args.ld_threshold
    log.record_parameter("LD_CLUMP_THRESHOLD", ld_threshold)

    ld_window_size = args.ld_window_size
    log.record_parameter("LD_WINDOW_SIZE", ld_window_size)

    exclude_ambiguous = args.exclude_ambiguous_alleles
    log.record_parameter("EXCLUDE_AMBIGUOUS_ALLELES", exclude_ambiguous)

    exclude_no_reference = args.exclude_no_reference
    log.record_parameter("EXCLUDE_NO_REFERENCE", exclude_no_reference)

    region = args.region
    log.record_parameter("REGION_INCLUDED", region)

    exclude_region = args.exclude_region
    log.record_parameter("REGION_EXCLUDED", exclude_region)

    # Parameters have been recorded so we initialize the logger.
    # This will make it in debug mode if needed and print the config values
    # to the screen.
    log.init_logger()

    # Read the summary statistics.
    log.logger.info("Reading summary statistics.")
    summary = read_summary_statistics(summary_filename, p_threshold, log,
                                      exclude_ambiguous=exclude_ambiguous,
                                      region=region,
                                      exclude_region=exclude_region)

    # Do the greedy variant selection.
    with geneparse.parsers["plink"](reference_filename) as reference:
        genotypes = InMemoryGenotypeExtractor(reference, summary.keys())

    try:
        grs = greedy_pick_clump(summary, genotypes, log)
    finally:
        genotypes.close()

    # Call the logger to dump the trace.
    log.log_selection_trace()

    # Write the selected GRS to file.
    write_selection(grs, log)

    # Display the execution time.
    t = time.time() - SCRIPT_START_TIME
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)

    log.logger.info(
        "Completed SNP selection in {:02d}:{:02d}:{:02d}."
        "".format(int(h), int(m), int(s))
    )
