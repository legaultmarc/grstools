import scipy.stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ...utils import parse_computed_grs_file


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
