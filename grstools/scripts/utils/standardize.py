from ...utils import parse_computed_grs_file


def standardize(args):
    out = args.out if args.out else "grs_standardized.csv"
    data = parse_computed_grs_file(args.grs_filename)

    data["grs"] = (data["grs"] - data["grs"].mean()) / data["grs"].std()
    data.to_csv(out)
