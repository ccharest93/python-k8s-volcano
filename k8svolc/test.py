import warnings
from typing import Any, Dict, Optional, Iterable, Mapping
from enum import Enum
from dataclasses import dataclass, field
from utils import macros, normalize_str, make_unique
LABEL_VERSION = "torchx.pytorch.org/version"
LABEL_APP_NAME = "torchx.pytorch.org/app-name"
LABEL_ROLE_INDEX = "torchx.pytorch.org/role-index"
LABEL_ROLE_NAME = "torchx.pytorch.org/role-name"
LABEL_REPLICA_ID = "torchx.pytorch.org/replica-id"
LABEL_KUBE_APP_NAME = "app.kubernetes.io/name"
LABEL_ORGANIZATION = "app.kubernetes.io/managed-by"
LABEL_UNIQUE_NAME = "app.kubernetes.io/instance"

RESERVED_MILLICPU = 100
RESERVED_MEMMB = 1024

ANNOTATION_ISTIO_SIDECAR = "sidecar.istio.io/inject"

LABEL_INSTANCE_TYPE = "node.kubernetes.io/instance-type"

class RetryPolicy(str, Enum):
    """
    Defines the retry policy for the ``Roles`` in the ``AppDef``.
    The policy defines the behavior when the role replica encounters a failure:

    1. unsuccessful (non zero) exit code
    2. hardware/host crashes
    3. preemption
    4. eviction

    .. note:: Not all retry policies are supported by all schedulers.
              However all schedulers must support ``RetryPolicy.APPLICATION``.
              Please refer to the scheduler's documentation for more information
              on the retry policies they support and behavior caveats (if any).

    1. REPLICA: Replaces the replica instance. Surviving replicas are untouched.
                Use with ``dist.ddp`` component to have torchelastic coordinate
                restarts and membership changes. Otherwise, it is up to the
                application to deal with failed replica departures and
                replacement replica admittance.
    2. APPLICATION: Restarts the entire application.

    """

    REPLICA = "REPLICA"
    APPLICATION = "APPLICATION"

RETRY_POLICIES: Mapping[str, Iterable[Mapping[str, str]]] = {
    RetryPolicy.REPLICA: [],
    RetryPolicy.APPLICATION: [
        {"event": "PodEvicted", "action": "RestartJob"},
        {"event": "PodFailed", "action": "RestartJob"},
    ],
}

@dataclass
class Resource:
    """
    Represents resource requirements for a ``Role``.

    Args:
        cpu: number of logical cpu cores. The definition of a CPU core depends
            on the scheduler. See your scheduler documentation for how a logical
            CPU core maps to physical cores and threads.
        gpu: number of gpus
        memMB: MB of ram
        capabilities: additional hardware specs (interpreted by scheduler)
        devices: a list of named devices with their quantities

    Note: you should prefer to use named_resources instead of specifying the raw
    resource requirement directly.
    """

    cpu: int
    gpu: int
    memMB: int
    capabilities: Dict[str, Any] = field(default_factory=dict)
    devices: Dict[str, int] = field(default_factory=dict)

    @staticmethod
    def copy(original: "Resource", **capabilities: Any) -> "Resource":
        """
        Copies a resource and applies new capabilities. If the same capabilities
        are present in the original resource and as parameter, the one from parameter
        will be used.
        """
        res_capabilities = dict(original.capabilities)
        res_capabilities.update(capabilities)
        return Resource(
            cpu=original.cpu,
            gpu=original.gpu,
            memMB=original.memMB,
            capabilities=res_capabilities,
            devices=original.devices,
        )
@dataclass
class BindMount:
    """
    Defines a bind mount to `mount --bind` a host path into the worker
    environment. See scheduler documentation on how bind mounts operate for each
    scheduler.

    Args:
        src_path: the path on the host
        dst_path: the path in the worker environment/container
        read_only: whether the mount should be read only
    """

    src_path: str
    dst_path: str
    read_only: bool = False

@dataclass
class VolumeMount:
    """
    Defines a persistent volume mount to mount into the worker environment.
    Args:
       src: the name or ID of the volume to mount
       dst_path: the path in the worker environment/container
       read_only: whether the mount should be read only
    """

    src: str
    dst_path: str
    read_only: bool = False

@dataclass
class DeviceMount:
    """
    Defines a host device to mount into the container.
    Args:
       src_path: the path on the host
       dst_path: the path in the worker environment/container
       permissions: the permissions to set on the device. Default: read, write, mknode
    """

    src_path: str
    dst_path: str
    permissions: str = "rwm"

def app_to_resource(
    app ,
    queue: str,
    service_account: Optional[str],
    priority_class: Optional[str] = None,
) -> Dict[str, object]:
    """
    app_to_resource creates a volcano job kubernetes resource definition from
    the provided AppDef. The resource definition can be used to launch the
    app on Kubernetes.

    To support macros we generate one task per replica instead of using the
    volcano `replicas` field since macros change the arguments on a per
    replica basis.

    Volcano has two levels of retries: one at the task level and one at the
    job level. When using the APPLICATION retry policy, the job level retry
    count is set to the minimum of the max_retries of the roles.
    """
    tasks = []
    unique_app_id = normalize_str(make_unique(app.name))
    for role_idx, role in enumerate(app.roles):
        for replica_id in range(role.num_replicas):
            values = macros.Values(
                img_root="",
                app_id=unique_app_id,
                replica_id=str(replica_id),
                rank0_env=f"VC_{normalize_str(app.roles[0].name)}_0_HOSTS".upper(),
            )
            if role_idx == 0 and replica_id == 0:
                values.rank0_env = "TORCHX_RANK0_HOST"
            name = normalize_str(f"{role.name}-{replica_id}")
            replica_role = values.apply(role)
            if role_idx == 0 and replica_id == 0:
                replica_role.env["TORCHX_RANK0_HOST"] = "localhost"

            pod = role_to_pod(name, replica_role, service_account)
            pod.metadata.labels.update(
                pod_labels(
                    app=app,
                    role_idx=role_idx,
                    role=role,
                    replica_id=replica_id,
                    app_id=unique_app_id,
                )
            )
            task: Dict[str, Any] = {
                "replicas": 1,
                "name": name,
                "template": pod,
            }
            if role.max_retries > 0:
                task["maxRetry"] = role.max_retries
                task["policies"] = RETRY_POLICIES[role.retry_policy]
                msg = f"""
Role {role.name} configured with restarts: {role.max_retries}. As of 1.4.0 Volcano
does NOT support retries correctly. More info: https://github.com/volcano-sh/volcano/issues/1651
                """
                warnings.warn(msg)
            if role.min_replicas is not None:
                # first min_replicas tasks are required, afterward optional
                task["minAvailable"] = 1 if replica_id < role.min_replicas else 0
            tasks.append(task)

    job_retries = min(role.max_retries for role in app.roles)
    job_spec = {
        "schedulerName": "volcano",
        "queue": queue,
        "tasks": tasks,
        "maxRetry": job_retries,
        "plugins": {
            # https://github.com/volcano-sh/volcano/issues/533
            "svc": ["--publish-not-ready-addresses"],
            "env": [],
        },
    }
    if priority_class is not None:
        job_spec["priorityClassName"] = priority_class

    resource: Dict[str, object] = {
        "apiVersion": "batch.volcano.sh/v1alpha1",
        "kind": "Job",
        "metadata": {"name": f"{unique_app_id}"},
        "spec": job_spec,
    }
    return resource

def role_to_pod(name: str, role, service_account: Optional[str]):
    from kubernetes.client.models import (  # noqa: F811 redefinition of unused
        V1Container,
        V1ContainerPort,
        V1EmptyDirVolumeSource,
        V1EnvVar,
        V1HostPathVolumeSource,
        V1ObjectMeta,
        V1PersistentVolumeClaimVolumeSource,
        V1Pod,
        V1PodSpec,
        V1ResourceRequirements,
        V1SecurityContext,
        V1Volume,
        V1VolumeMount,
    )

    # limits puts an upper cap on the resources a pod may consume.
    # requests is how much the scheduler allocates. We assume that the jobs will
    # be allocation the whole machine so requests is slightly lower than the
    # requested resources to account for the Kubernetes node reserved resources.
    limits = {}
    requests = {}

    resource = role.resource
    if resource.cpu > 0:
        mcpu = int(resource.cpu * 1000)
        limits["cpu"] = f"{mcpu}m"
        request_mcpu = max(mcpu - RESERVED_MILLICPU, 0)
        requests["cpu"] = f"{request_mcpu}m"
    if resource.memMB > 0:
        limits["memory"] = f"{int(resource.memMB)}M"
        request_memMB = max(int(resource.memMB) - RESERVED_MEMMB, 0)
        requests["memory"] = f"{request_memMB}M"
    if resource.gpu > 0:
        requests["nvidia.com/gpu"] = limits["nvidia.com/gpu"] = str(resource.gpu)

    for device_name, device_limit in resource.devices.items():
        limits[device_name] = str(device_limit)

    resources = V1ResourceRequirements(
        limits=limits,
        requests=requests,
    )

    node_selector: Dict[str, str] = {}
    if LABEL_INSTANCE_TYPE in resource.capabilities:
        node_selector[LABEL_INSTANCE_TYPE] = resource.capabilities[LABEL_INSTANCE_TYPE]

    # To support PyTorch dataloaders we need to set /dev/shm to larger than the
    # 64M default so we mount an unlimited sized tmpfs directory on it.
    SHM_VOL = "dshm"
    volumes = [
        V1Volume(
            name=SHM_VOL,
            empty_dir=V1EmptyDirVolumeSource(
                medium="Memory",
            ),
        ),
    ]
    volume_mounts = [
        V1VolumeMount(name=SHM_VOL, mount_path="/dev/shm"),
    ]
    security_context = V1SecurityContext()

    for i, mount in enumerate(role.mounts):
        mount_name = f"mount-{i}"
        if isinstance(mount, BindMount):
            volumes.append(
                V1Volume(
                    name=mount_name,
                    host_path=V1HostPathVolumeSource(
                        path=mount.src_path,
                    ),
                )
            )
            volume_mounts.append(
                V1VolumeMount(
                    name=mount_name,
                    mount_path=mount.dst_path,
                    read_only=mount.read_only,
                )
            )
        elif isinstance(mount, VolumeMount):
            volumes.append(
                V1Volume(
                    name=mount_name,
                    persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                        claim_name=mount.src,
                    ),
                )
            )
            volume_mounts.append(
                V1VolumeMount(
                    name=mount_name,
                    mount_path=mount.dst_path,
                    read_only=mount.read_only,
                )
            )
        elif isinstance(mount, DeviceMount):
            volumes.append(
                V1Volume(
                    name=mount_name,
                    host_path=V1HostPathVolumeSource(
                        path=mount.src_path,
                    ),
                )
            )
            volume_mounts.append(
                V1VolumeMount(
                    name=mount_name,
                    mount_path=mount.dst_path,
                    read_only=(
                        "w" not in mount.permissions and "m" not in mount.permissions
                    ),
                )
            )
            security_context.privileged = True
        else:
            raise TypeError(f"unknown mount type {mount}")

    container = V1Container(
        command=[role.entrypoint] + role.args,
        image=role.image,
        name=name,
        env=[
            V1EnvVar(
                name=name,
                value=value,
            )
            for name, value in role.env.items()
        ],
        resources=resources,
        ports=[
            V1ContainerPort(
                name=name,
                container_port=port,
            )
            for name, port in role.port_map.items()
        ],
        volume_mounts=volume_mounts,
        security_context=security_context,
    )

    return V1Pod(
        spec=V1PodSpec(
            containers=[container],
            restart_policy="Never",
            service_account_name=service_account,
            volumes=volumes,
            node_selector=node_selector,
        ),
        metadata=V1ObjectMeta(
            annotations={
                # Disable the istio sidecar as it prevents the containers from
                # exiting once finished.
                ANNOTATION_ISTIO_SIDECAR: "false",
            },
            labels={},
        ),
    )


def pod_labels(
    app, role_idx: int, role, replica_id: int, app_id: str
) -> Dict[str, str]:
    return {
        LABEL_APP_NAME: app.name,
        LABEL_ROLE_INDEX: str(role_idx),
        LABEL_ROLE_NAME: role.name,
        LABEL_REPLICA_ID: str(replica_id),
        LABEL_KUBE_APP_NAME: app.name,
        LABEL_ORGANIZATION: "torchx.pytorch.org",
        LABEL_UNIQUE_NAME: app_id,
    }

_MOUNT_OPT_MAP: Mapping[str, str] = {
    "type": "type",
    "destination": "dst",
    "dst": "dst",
    "target": "dst",
    "read_only": "readonly",
    "readonly": "readonly",
    "source": "src",
    "src": "src",
    "perm": "perm",
}
from typing import List, Union

def parse_mounts(opts: List[str]) -> List[Union[BindMount, VolumeMount, DeviceMount]]:
    """
    parse_mounts parses a list of options into typed mounts following a similar
    format to Dockers bind mount.

    Multiple mounts can be specified in the same list. ``type`` must be
    specified first in each.

    Ex:
        type=bind,src=/host,dst=/container,readonly,[type=bind,src=...,dst=...]

    Supported types:
        BindMount: type=bind,src=<host path>,dst=<container path>[,readonly]
        VolumeMount: type=volume,src=<name/id>,dst=<container path>[,readonly]
        DeviceMount: type=device,src=/dev/<dev>[,dst=<container path>][,perm=rwm]
    """
    mount_opts = []
    cur = {}
    for opt in opts:
        key, _, val = opt.partition("=")
        if key not in _MOUNT_OPT_MAP:
            raise KeyError(
                f"unknown mount option {key}, must be one of {list(_MOUNT_OPT_MAP.keys())}"
            )
        key = _MOUNT_OPT_MAP[key]
        if key == "type":
            cur = {}
            mount_opts.append(cur)
        elif len(mount_opts) == 0:
            raise KeyError("type must be specified first")
        cur[key] = val

    mounts = []
    for opts in mount_opts:
        typ = opts.get("type")
        if typ == MountType.BIND:
            mounts.append(
                BindMount(
                    src_path=opts["src"],
                    dst_path=opts["dst"],
                    read_only="readonly" in opts,
                )
            )
        elif typ == MountType.VOLUME:
            mounts.append(
                VolumeMount(
                    src=opts["src"], dst_path=opts["dst"], read_only="readonly" in opts
                )
            )
        elif typ == MountType.DEVICE:
            src = opts["src"]
            dst = opts.get("dst", src)
            perm = opts.get("perm", "rwm")
            for c in perm:
                if c not in "rwm":
                    raise ValueError(
                        f"{c} is not a valid permission flags must one of r,w,m"
                    )
            mounts.append(DeviceMount(src_path=src, dst_path=dst, permissions=perm))
        else:
            valid = list(str(item.value) for item in MountType)
            raise ValueError(f"invalid mount type {repr(typ)}, must be one of {valid}")
    return mounts

class MountType(str, Enum):
    BIND = "bind"
    VOLUME = "volume"
    DEVICE = "device"
NULL_RESOURCE: Resource = Resource(cpu=-1, gpu=-1, memMB=-1)


# no-arg static factory method to use with default_factory in @dataclass
# needed to support python 3.11 since mutable defaults for dataclasses are not allowed in 3.11
MISSING: str = "<MISSING>"
def _null_resource() -> Resource:
    return NULL_RESOURCE
@dataclass
class Role:
    """
    A set of nodes that perform a specific duty within the ``AppDef``.
    Examples:

    1. Distributed data parallel app - made up of a single role (trainer).

    2. App with parameter server - made up of multiple roles (trainer, ps).

    .. note:: An ``image`` is a software bundle that is installed on the container
              scheduled by the scheduler. The container on the scheduler dictates
              what an image actually is. An image could be as simple as a tar-ball
              or map to a docker image. The scheduler typically knows how to "pull"
              the image given an image name (str), which could be a simple name
              (e.g. docker image) or a url e.g. ``s3://path/my_image.tar``).

    Usage:

    ::

     trainer = Role(name="trainer",
                    image = "pytorch/torch:1",
                    entrypoint = "my_trainer.py"
                    args = ["--arg", "foo", ENV_VAR="FOOBAR"],
                    num_replicas = 4,
                    resource = Resource(cpu=1, gpu=1, memMB=500),
                    port_map={"tcp_store":8080, "tensorboard": 8081},
                    metadata={"local_cwd.property", value})

    Args:
            name: name of the role
            image: a software bundle that is installed on a container.
            entrypoint: command (within the container) to invoke the role
            args: commandline arguments to the entrypoint cmd
            env: environment variable mappings
            num_replicas: number of container replicas to run
            min_replicas: minimum number of replicas for the job to start. When
                set the job size can automatically adjust between min_replicas
                and num_replicas depending on the cluster resources and
                policies. If the scheduler doesn't support auto scaling this
                field is ignored and the job size will be num_replicas.
            max_retries: max number of retries before giving up
            retry_policy: retry behavior upon replica failures
            resource: Resource requirement for the role. The role should be scheduled
                by the scheduler on ``num_replicas`` container, each of them should have at
                least ``resource`` guarantees.
            port_map: Port mapping for the role. The key is the unique identifier of the port
                e.g. "tensorboard": 9090
            metadata: Free form information that is associated with the role, for example
                scheduler specific data. The key should follow the pattern: ``$scheduler.$key``
            mounts: a list of mounts on the machine
    """

    name: str
    image: str
    min_replicas: Optional[int] = None
    base_image: Optional[str] = None  # DEPRECATED DO NOT SET, WILL BE REMOVED SOON
    entrypoint: str = MISSING
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    num_replicas: int = 1
    max_retries: int = 0
    retry_policy: RetryPolicy = RetryPolicy.APPLICATION
    resource: Resource = field(default_factory=_null_resource)
    port_map: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    mounts: List[Union[BindMount, VolumeMount, DeviceMount]] = field(
        default_factory=list
    )


class AppDef:
    def __init__(self, role) -> None:  
        self.name = "dist_app"
        self.roles = [role]