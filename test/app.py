import argparse


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("app", add_help=add_help)
    parser.add_argument("--name", default="", help="path to config file", required=True)
    return parser


def main(args):
    print(args)
    print(f"Hello, {args.name}!")


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
