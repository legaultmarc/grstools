import logging

import matplotlib.pyplot as plt

from ...utils import parse_computed_grs_file


logger = logging.getLogger(__name__)


def histogram(args):
    """Generate a histogram of GRS values."""
    out = args.out if args.out else "grs_histogram.png"
    data = parse_computed_grs_file(args.grs_filename)

    plt.hist(data["grs"], bins=args.bins)
    plt.xlabel("GRS")
    logger.info("WRITING histogram to file '{}'.".format(out))

    if out.endswith(".png"):
        plt.savefig(out, dpi=300)
    else:
        plt.savefig(out)
