import k8svolc.k8svolc as k8svolc
import logging
import sys
import os
from test.app import get_args_parser as get_entrypoint_args_parser
from test.app import main as entrypoint

logger = logging.getLogger("app-launcher")


def main():
    # ARGPARSING FROM ENTRYPOINT AND LAUNCHER
    description = "Example launcher"
    entrypoint_parser = get_entrypoint_args_parser(add_help=False)
    parents = [entrypoint_parser]
    args_parser = k8svolc.get_args_parser(description=description, parents=parents)
    args = args_parser.parse_args()

    # SETUP LOGGING for launcher
    ### read more on kubernetes centralized logging solutions

    # EXTRA ARG CHECKS
    entrypoint(args)  # build container, push to repo, then submit job to volcano


if __name__ == "__main__":
    sys.exit(main())
