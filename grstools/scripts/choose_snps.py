"""
Choose SNPs from GWAS summary statistics.

Method 1:
    Sort by significance.
    Choose top SNP.
    Remove all SNPs in LD.
    Loop until p-value threshold or n variants is reached.

Ideas:
    - Skew GRS towards only biomarker increasing or decreasing alleles.

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
import bisect
import pickle
import datetime
import time
import os
import csv
import collections

import geneparse
from geneparse.exceptions import InvalidChromosome

import numpy as np

from ..utils import parse_grs_file, compute_ld
from ..version import grstools_version


logger = logging.getLogger(__name__)


class ParameterObject(object):
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
    @staticmethod
    def valid(o):
        """Validate that a str is of the form chrXX:START-END."""
        try:
            _parse_region(o)
        except ValueError:
            return False
        return True


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

        start = variant.pos - self.parameters["LD_WINDOW_SIZE"] // 2
        end = variant.pos + self.parameters["LD_WINDOW_SIZE"] // 2

        self.logger.debug(
            "\tLD_REGION {} to {} ({}: {}-{}) [{} candidates]"
            "".format(other_loci[0], other_loci[-1], variant.chrom,
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


def region_query(index, variant, padding):
    """Return the index of the elements defining the genomic window of size
        'padding' around the Variant's position.

    Args:
        index (dict): Dict of chromosomes to sorted list of positions. The
            positions correspond to all the variants with available summary
            statistics.
        variant (Variant): The currently considered variant.
        padding (int): The size of the genomic window.

    Returns:
        tuple[int, int]: The **index** of the elements defining the window
            boundaries.

    """
    index = index[variant.chrom]
    left = bisect.bisect(index, variant.pos - padding // 2)
    right = bisect.bisect(index, variant.pos + padding // 2)
    return left, right


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


def build_index(variants):
    """Build an index for genomic range queries.

    Args:
        Iterable[Variant]: An iterable of variant instances.

    Returns:
        dict[str, List[Variant]]: A dict mapping chromosomes to lists of
            available Variants sorted by position.

    """
    index = collections.defaultdict(list)

    # Start by separating by chromosome.
    for v in variants:
        index[v.chrom].append(v)

    # Sort by position
    for chrom in index.keys():
        index[chrom] = sorted(index[chrom], key=lambda x: x.pos)

    return index


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
        region = _parse_region(region)

    if exclude_region is not None:
        exclude_region = _parse_region(exclude_region)

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


def extract_genotypes(filename, summary, maf_threshold, log):
    """Extract genotypes from a plink file (most likely reference panel).

    These genotypes are used for LD computations by the greedy_pick_clump
    algorithm.

    Args:
        filename (str): The path to the plink prefix
            (`example`{.bed,.bim,.fam}).
        summary (Dict[Variant, Row]): Information from the summary statistics
            file. Row contains fields like effect, reference, risk, p_value,
            etc.
        maf_threshold (float): Skip variants with a MAF under the provided
            threshold. The information from the summary statistics file
            is used and if it is unavailable, the MAF is computed from the
            reference panel.
        log (SNPSelectionLog): An object used to keep track of the selection
            process.

    Returns:
        Dict[Variant, np.array]: The dict will contain the variants from the
            'summary' dict that could be found in the provided plink file.
            **The np.array will be a numpy vector of sample-standardized
            genotypes.**

    A warning will be displayed if no genotype data is available. If that is
    the case, the variant will be excluded (i.e. not selected for inclusion in
    the GRS).

    """
    genotypes = {}

    # Extract the genotypes for all the variants in the summary.
    reference = geneparse.parsers["plink"](filename)

    for variant, stats in summary.items():
        # Check if MAF is already known.
        if stats.maf is not None and stats.maf < maf_threshold:
            log.record_excluded(
                variant,
                "MAF_FILTER",
                "MAF recorded as {:g} in summary statistics file"
                "".format(stats.maf)
            )
            continue

        ref_geno = reference.get_variant_genotypes(variant)

        if len(ref_geno) == 0:
            pass

        elif len(ref_geno) == 1:
            g = ref_geno[0].genotypes

            # Compute the maf.
            mean = np.nanmean(g)
            maf = mean / 2
            stats.maf = min(maf, 1 - maf)

            if maf >= maf_threshold:
                # Standardize.
                g = (g - mean) / np.nanstd(g)
                genotypes[variant] = g

            else:
                log.record_excluded(
                    variant,
                    "MAF_FILTER",
                    "MAF from reference panel is {:g}".format(maf)
                )

        else:
            log.record_excluded(
                variant,
                "DUP_OR_MULTIALLELIC_IN_REFERENCE_PANEL"
            )

    log.logger.info(
        "Extracted {} variants from the reference (variants missing from the "
        "reference and variants filtered based on MAF will be logged)"
        "".format(len(genotypes))
    )

    return genotypes


def build_genotype_matrix(cur, loci, genotypes, summary):
    """Build the genotype matrix of neighbouring variants.

    Args:
        cur (Variant): The Variant currently being considered.
        loci (List[Variant]): List of variants in a genomic window around cur.
        genotypes (Dict[Variant, np.array]): Genotype for variants contained
            in the reference panel.
        summary (Dict[Variant, Row]): Information on the variants that have
            not been selected yet.

    Returns:
        Tuple[np.array, List[Variant]]: The tuple containes a genotype matrix
        of size n_samples x n_variants and a list of variants corresponding to
        the columns of the genotype matrix.

    """
    # other_genotypes is a genotype matrix
    # retained_loci is a list of variants
    other_genotypes = []
    retained_loci = []

    for locus in loci:
        # Get the genotypes in the reference.
        geno = genotypes[locus]

        if locus == cur:
            continue

        # Variant was already excluded.
        if locus not in summary:
            continue

        other_genotypes.append(geno)
        retained_loci.append(locus)

    other_genotypes = np.array(other_genotypes).T

    return other_genotypes, retained_loci


def greedy_pick_clump(summary, genotypes, index, ld_threshold, ld_window_size,
                      log, target_n=None):
    """Greedy algorithm to select SNPs for inclusion in the GRS.

    Args:
        summary (Dict[Variant, Row]): Dict representation of the summary
            statistics file containing Variants as keys and their information
            as values.
        genotypes (Dict[Variant, np.array]): Individual level standardized
            genotypes from a reference panel for LD computation.
        index (Dict[str, List[Variant]): The keys are the chromosomes and the
            values are lists of position-sorted variants from the summary
            statistics file. This is used to lookup variants by region when
            selecting blocks of possibly correlated variants.
        ld_threshold (float): Maximum allowed LD between variants included in
            the GRS. Large values could lead to the inclusion of correlated
            variants in the GRS whereas small values could discard independant
            variants because of spurious correlation.
        ld_window_size (float): When LD-clumping variants, neighbouring
            variants in a genomic window are selected. This is the size of
            that window. Larger window sizes are safer but slower.
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

    # Extract the positions from the index to comply with the bisect API.
    index_positions = {}
    for chrom in index:
        index_positions[chrom] = [i.pos for i in index[chrom]]

    excluded_variants = log.get_excluded_variants()

    while len(summary) > 0:
        # One of the stop conditions is target n, which we check.
        if target_n is not None and len(out) >= target_n:
            log.logger.info("Target number of variants reached.")
            break

        # Get the next best variant.
        cur, info = summary.popitem(last=False)
        cur_geno = None  # Reset the genotypes.

        # If the variant was excluded, continue.
        if cur in excluded_variants:
            continue

        # Make sure genotypes are available and log for debugging the current
        # variant.
        available_genotypes = cur in genotypes

        if not available_genotypes:
            if log.parameters["EXCLUDE_NO_REFERENCE"]:
                log.record_excluded(cur, "NOT_IN_REFERENCE_PANEL")
                continue

            else:
                log.record_included_special(
                    cur,
                    "NOT_IN_REFERENCE",
                    "Variant {} was absent from the reference panel but was "
                    "still included in the GRS. It is important to validate "
                    "that it is not correlated with other included variants."
                    "".format(cur)
                )

        else:
            cur_geno = genotypes[cur]

        # Add it to the GRS.
        log.record_included(cur)
        out.append(info)

        if cur_geno is None:
            continue

        else:
            # We specifically check for MAF close to 0.5 and ambiguous alleles
            # to warn the user.
            assert info.maf <= 0.5
            if cur.alleles_ambiguous() and info.maf >= 0.4:
                log.record_included_special(
                    cur,
                    "INCLUDED_AMBIGUOUS",
                    "Variant {} was included in the GRS and could be strand "
                    "ambiguous (alleles: {} and MAF: {:.2f})"
                    "".format(cur, cur.alleles, info.maf)
                )

        # Do a region query in the index to get neighbouring variants.
        left, right = region_query(index_positions, cur, ld_window_size)

        # We got the indices from index_positions (which correspond with
        # index). So we can get the Variant instances in 'index'.
        loci = index[cur.chrom][left:right]

        # Extract genotypes.
        other_genotypes, retained_loci = build_genotype_matrix(
            cur, loci, genotypes, summary
        )

        if len(retained_loci) == 0:
            r2 = []
        else:
            # Compute the LD between all the neighbouring variants and the
            # current variant.
            r2 = compute_ld(cur_geno, other_genotypes, r2=True)

        log.record_ld_block(cur, retained_loci, r2)

        # Remove all the correlated variants.
        for variant, pair_ld in zip(retained_loci, r2):
            if pair_ld > ld_threshold:
                log.record_excluded(
                    variant,
                    "LD_CLUMPED",
                    "LD with {} {} is {:g}".format(
                        cur.name if cur.name else "",
                        cur, pair_ld
                    )
                )

                del summary[variant]

    return out


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

    # Get the genotypes from the reference.
    log.logger.info("Extracting genotypes.")
    try:
        previous_geneparse_log = geneparse.config.LOG_NOT_FOUND
        geneparse.config.LOG_NOT_FOUND = False
        genotypes = extract_genotypes(reference_filename, summary,
                                      maf_threshold, log)
    finally:
        geneparse.config.LOG_NOT_FOUND = previous_geneparse_log

    # Build an index for the available variants.
    index = build_index(set(genotypes.keys()))

    # Do the greedy variant selection.
    grs = greedy_pick_clump(summary, genotypes, index, ld_threshold,
                            ld_window_size, log, target_n)

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
