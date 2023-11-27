import argparse
from typing import List, Optional, Tuple
import re
import sys
import os
import logging
from test import Resource, parse_mounts, Role, app_to_resource, AppDef
from utils import macros, _TORCH_DEBUG_FLAGS, _noquote, print_push_events, _args_join, BLUE, ENDC, GRAY

import docker
from kubernetes import client, config

logger: logging.Logger = logging.getLogger(__name__)

def DDP_launch(args, unparsed_entrypoint_args):
    logging.basicConfig(
        level=args.log_level,
        format=f"{BLUE}torchx{ENDC} {GRAY}%(asctime)s %(levelname)-8s{ENDC} %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    role = Role(
                name=unparsed_entrypoint_args.split(" ")[0],
                image=args.image_repo,
                min_replicas=args.min_nnodes,
                entrypoint="bash",
                num_replicas=int(args.max_nnodes),
                resource=Resource(cpu=args.cpu, gpu=args.gpu, memMB=args.memMB),
                args=["-c", _args_join(args.cmd)],
                env=args.env,
                port_map={
                    "c10d": args.rdzv_port,
                },
                max_retries=args.max_retries,
                mounts=parse_mounts(args.mounts) if args.mounts else [],
    )
    app = AppDef(role)
    workspace = os.path.abspath(os.path.dirname(sys.argv[0]))

    #DOCKER image building
    docker_client = docker.from_env()
    old_image = ''
    try:
        image = docker_client.images.pull(args.image_repo, all_tags=True)
        old_image = image[0].id[7:]
    except Exception as e:
        logger.warning(
            f"failed to pull image {role.image}, falling back to local: {e}"
        )
    image, _ =docker_client.images.build(path=workspace,  rm=True)
    role.image = image.id[7:]
    if old_image != role.image:
        logger.info(
            f"Built new image `{role.image}` based on Dockerfile in `{workspace}"
            f" and changes in workspace `{workspace}` for role[0]={role.name}."
        )
        image.tag(repository=args.image_repo, tag=role.image)
        print_push_events(docker_client.images.push(repository= args.image_repo,tag=role.image, stream=True, decode=True))
    else:
        logger.info(
            f"Reusing original image `{old_image}` for role[0]={role.name}."
            " Either a patch was built or no changes to workspace was detected."
        )
    role.image = args.image_repo + ":" + role.image


    #KUBERNETES vcjob pushing
    resource = app_to_resource(app, args.queue, None, None)
    config.load_kube_config()
    k8s_client = client.CustomObjectsApi()
    resp = k8s_client.create_namespaced_custom_object(
    group="batch.volcano.sh",
    version="v1alpha1",
    namespace=args.queue,
    plural="jobs",
    body=resource,
    )
    app_handle = f'kubernetes://k8svolc/{args.queue}:{resp["metadata"]["name"]}'
    logger.info(f"Launched app: {app_handle}")

#PARSING FUNCTIONS FOR DDP
def DDP_parse_args(args, entrypoint_args_parser =None) -> argparse.Namespace:
    try:
        launcher_parser = DDP_get_args_parser()
        launcher_args = sys.argv[1:sys.argv.index("--")]
        launcher_args = launcher_parser.parse_args(launcher_args)
        launcher_args.min_nnodes, launcher_args.max_nnodes, launcher_args.nproc_per_node, launcher_args.nnodes_rep = parse_nnodes(launcher_args.j)
        if launcher_args.max_nnodes == 1:
        # using port 0 makes elastic chose a free random port which is ok
        # for single-node jobs since all workers run under a single agent
        # When nnodes is 0 and max_nnodes is 1, it's stil a single node job
        # but pending until the resources become available
            launcher_args.rdzv_endpoint = "localhost:0"
        else:
            # for multi-node, rely on the rank0_env environment variable set by
            # the schedulers (see scheduler implementation for the actual env var this maps to)
            # some schedulers (e.g. aws batch) make the rank0's ip-addr available on all BUT on rank0
            # so default to "localhost" if the env var is not set or is empty
            # rdzv_endpoint bash resolves to something to the effect of
            # ${TORCHX_RANK0_HOST:=localhost}:29500
            # use $$ in the prefix to escape the '$' literal (rather than a string Template substitution argument)
            launcher_args.rdzv_endpoint = _noquote(f"$${{{macros.rank0_env}:=localhost}}:{launcher_args.rdzv_port}")
        if launcher_args.env is None:
            launcher_args.env = {}
    
        if len(sys.argv) == sys.argv.index("--")+1:
            raise ValueError("No entrypoint arguments given")
        if len(sys.argv) > sys.argv.index("--")+2:
            entrypoint_args = sys.argv[sys.argv.index("--")+2:]
            if entrypoint_args_parser is not None:
                entrypoint_args_parser.parse_args(entrypoint_args)
        unparsed_entrypoint_args = " ".join(sys.argv[sys.argv.index("--")+1:])
        launcher_args.env.setdefault("LOGLEVEL", os.getenv("LOGLEVEL", "WARNING"))
        if launcher_args.debug:
            launcher_args.env.update(_TORCH_DEBUG_FLAGS)
        launcher_args.cmd = [
            "torchrun",
            "--rdzv_backend",
            launcher_args.rdzv_backend,
            "--rdzv_endpoint",
            launcher_args.rdzv_endpoint,
            "--rdzv_id",
            f"{macros.app_id}",
            "--nnodes",
            launcher_args.nnodes_rep,
            "--nproc_per_node",
            str(launcher_args.nproc_per_node),
            "--tee",
            "3",
            "--role",
            "",
        ]
        launcher_args.cmd += unparsed_entrypoint_args.split(" ")

    except:
        print(launcher_parser.print_usage())
        sys.exit(1)
    

    return launcher_args, unparsed_entrypoint_args

def DDP_get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
) -> argparse.ArgumentParser:
    parents = parents or []
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--log_level",
        type=str,
        help="Python logging log level",
        default=os.getenv("LOGLEVEL", "INFO"),
    )
    parser.add_argument(
        "--image_repo",
        type=str,
        help="Image repository to push built image/ for the job to query"
    )
    parser.add_argument(
        "--queue",
        default="default",
        type=str,
        help="Volcano queue where to submit",
    )
    parser.add_argument(
        "--j",
        default="1x1",
        type=str,
        help="[min_nnodes:]nnodesxnproc_per_node, for gpu hosts, nproc_per_node must not exceed num gpus",
    )
    parser.add_argument(
        "--mounts",
        default=None,
        type=str,
        help="mounts to mount into the worker environment/container (ex. type=<bind/volume>,src=/host,dst=/job[,readonly]).\nSee scheduler documentation for more info.",
    )
    parser.add_argument(
        "--cpu",
        default=2,
        type=int,
        help="number of cpus per replica",
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="number of gpus per replica",
    )
    parser.add_argument(
        "--memMB",
        default=1024,
        type=int,
        help="memory in MB per replica",
    )
    parser.add_argument(
        "--env",
        default=None,
        type=str,
        help="environment varibles to be passed to the run (e.g. ENV1=v1,ENV2=v2,ENV3=v3)",
    )
    parser.add_argument(
        "--rdzv_port",
        default=29500,
        type=int,
        help="the port on rank0's host to use for hosting the c10d store used for rendezvous.\nOnly takes effect when running multi-node. When running single node, this parameter\nis ignored and a random free port is chosen.",
    )
    parser.add_argument(
        "--rdzv_backend",
        default="c10d",
        type=str,
        help="the rendezvous backend to use. Only takes effect when running multi-node.",
    )
    parser.add_argument(
        "--max_retries",
        default=0,
        type=int,
        help="the number of scheduler retries allowed",
    )
    parser.add_argument(
        "--debug",
        default='False',
        action="store_true",
        help="whether to run with preset debug flags enabled",
    )
    return parser

def parse_nnodes(j: str) -> Tuple[int, int, int, str]:
    """
    parse_nnodes converts a node and process string into the individual
    components. Format is ``[[<min_replicas>:]<replicas>x]<num processes>``.
    """
    if re.match("^\\d+:\\d+x\\d+$", j):  # match 2:4x1
        nnodes_rep, nproc_per_node = j.split("x")
        min_nnodes, max_nnodes = nnodes_rep.split(":")
    elif re.match("^\\d+x\\d+$", j):  # match 2x1
        min_nnodes, nproc_per_node = j.split("x")
        max_nnodes = min_nnodes
        nnodes_rep = min_nnodes
    elif re.match("^\\d+$", j):  # match 2
        min_nnodes = "1"
        max_nnodes = min_nnodes
        nnodes_rep = min_nnodes
        nproc_per_node = j
    else:
        raise ValueError(
            f"Invalid format for -j, usage example: 1:2x4 or 1x4 or 4. Given: {j}"
        )
    return int(min_nnodes), int(max_nnodes), int(nproc_per_node), nnodes_rep