import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # datasets parameters
    parser.add_argument("--dataset", type=str, default=moscoco,
                        help="choose which dataset?")

    args = parser.parse_args()
    return args