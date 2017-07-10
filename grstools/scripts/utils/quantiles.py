import logging

import numpy as np

from ...utils import parse_computed_grs_file


logger = logging.getLogger(__name__)


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
