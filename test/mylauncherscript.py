import torchxk8s
import logging
import sys
import os
from test.mymodelcode import get_args_parser as get_entrypoint_args_parser

logger = logging.getLogger("example-launcher")


def main():
    # ARGPARSING FROM ENTRYPOINT AND LAUNCHER
    description = "Example launcher"
    entrypoint_parser = get_entrypoint_args_parser(add_help=False)
    parents = [entrypoint_parser]
    args_parser = torchxk8s.get_args_parser(description=description, parents=parents)
    args = args_parser.parse_args()

    # SETUP LOGGING for launcher
    ### read more on kubernetes centralized logging solutions

    # EXTRA ARG CHECKS
    assert os.path.exists(args.config_file), "Configuration file does not exist!"
    torchxk8s.submitjob()  # build container, push to repo, then submit job to volcano


if __name__ == "__main__":
    sys.exit(main())
