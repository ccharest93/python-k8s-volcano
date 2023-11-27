import k8svolc
import logging
import sys

def main():
    #Split argument list to launcher and entrypoint
    launcher_args, unparsed_entrypoint_args =k8svolc.DDP_parse_args(sys.argv)

    #ADDITIONAL CHECKS ON THE ENTRYPOINT ARGUMENTS

    # SETUP LOGGING for launcher
    ### read more on kubernetes centralized logging solutions

    #LAUNCH
    k8svolc.DDP_launch(launcher_args, unparsed_entrypoint_args)



if __name__ == "__main__":
    sys.exit(main())
