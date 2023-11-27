import re
import os
import random
import struct
from dataclasses import dataclass, asdict
import copy
from string import Template
from typing import Dict, Iterable, TextIO
import shlex
import sys

START_CANDIDATES: str = "bcdfghjklmnpqrstvwxz"
END_CANDIDATES: str = START_CANDIDATES + "012345679"

class macros:
    """
    Defines macros that can be used in the elements of ``Role.args``
    values of ``Role.env``. The macros will be substituted at runtime
    to their actual values.

    .. warning:: Macros used fields of :py:class:`Role` other than the ones
                 mentioned above, are NOT substituted.

    Available macros:

    1. ``img_root`` - root directory of the pulled container.image
    2. ``app_id`` - application id as assigned by the scheduler
    3. ``replica_id`` - unique id for each instance of a replica of a Role,
                        for instance a role with 3 replicas could have the 0, 1, 2
                        as replica ids. Note that when the container fails and is
                        replaced, the new container will have the same ``replica_id``
                        as the one it is replacing. For instance if node 1 failed and
                        was replaced by the scheduler the replacing node will also
                        have ``replica_id=1``.

    Example:

    ::

     # runs: hello_world.py --app_id ${app_id}
     trainer = Role(
                name="trainer",
                entrypoint="hello_world.py",
                args=["--app_id", macros.app_id],
                env={"IMAGE_ROOT_DIR": macros.img_root})
     app = AppDef("train_app", roles=[trainer])
     app_handle = session.run(app, scheduler="local_docker", cfg={})

    """

    img_root = "${img_root}"
    base_img_root = "${base_img_root}"
    app_id = "${app_id}"
    replica_id = "${replica_id}"

    # rank0_env will be filled with the name of the environment variable that
    # provides the master host address. To get the actual hostname the
    # environment variable must be resolved by the app via either shell
    # expansion (wrap sh/bash) or via the application.
    # This may not be available on all schedulers.
    rank0_env = "${rank0_env}"

    @dataclass
    class Values:
        img_root: str
        app_id: str
        replica_id: str
        rank0_env: str
        base_img_root: str = "DEPRECATED"

        def apply(self, role):
            """
            apply applies the values to a copy the specified role and returns it.
            """

            role = copy.deepcopy(role)
            role.args = [self.substitute(arg) for arg in role.args]
            role.env = {key: self.substitute(arg) for key, arg in role.env.items()}
            return role

        def substitute(self, arg: str) -> str:
            """
            substitute applies the values to the template arg.
            """
            return Template(arg).safe_substitute(**asdict(self))

def get_len_random_id(string_length: int) -> str:
    """
    Generates an alphanumeric string ID that matches the requirements from
    https://kubernetes.io/docs/concepts/overview/working-with-objects/names/
    """
    out = ""
    for i in range(string_length):
        if out == "":
            candidates = START_CANDIDATES
        else:
            candidates = END_CANDIDATES

        out += random.choice(candidates)

    return out

def random_uint64() -> int:
    """
    random_uint64 returns an random unsigned 64 bit int.
    """
    return struct.unpack("!Q", os.urandom(8))[0]

def random_id() -> str:
    """
    Generates an alphanumeric string ID that matches the requirements from
    https://kubernetes.io/docs/concepts/overview/working-with-objects/names/
    """
    out = ""
    v = random_uint64()
    while v > 0:
        if out == "":
            candidates = START_CANDIDATES
        else:
            candidates = END_CANDIDATES

        char = v % len(candidates)
        v = v // len(candidates)
        out += candidates[char]
    return out

def make_unique(name: str, string_length: int = 0) -> str:
    """
    Appends a unique 64-bit string to the input argument.

    Returns:
        string in format $name-$unique_suffix
    """
    return (
        f"{name}-{random_id()}"
        if string_length == 0
        else f"{name}-{get_len_random_id(string_length)}"
    )

def normalize_str(data: str) -> str:
    """
    Invokes ``lower`` on thes string and removes all
    characters that do not satisfy ``[a-z0-9\\-]`` pattern.
    This method is mostly used to make sure kubernetes and gcp_batch scheduler gets
    the job name that does not violate its restrictions.
    """
    if data.startswith("-"):
        data = data[1:]
    pattern = r"[a-z0-9\-]"
    return "".join(re.findall(pattern, data.lower()))


# only print colors if outputting directly to a terminal
if not sys.stdout.closed and sys.stdout.isatty():
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    ORANGE = "\033[38:2:238:76:44m"
    GRAY = "\033[2m"
    ENDC = "\033[0m"
else:
    GREEN = ""
    ORANGE = ""
    BLUE = ""
    GRAY = ""
    ENDC = ""

_TORCH_DEBUG_FLAGS: Dict[str, str] = {
    "CUDA_LAUNCH_BLOCKING": "1",
    "NCCL_DESYNC_DEBUG": "1",
    "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
}


class _noquote(str):
    """
    _noquote is a wrapper around str that indicates that the argument shouldn't
    be passed through shlex.quote.
    """

    pass

def print_push_events(
    events: Iterable[Dict[str, str]],
    stream: TextIO = sys.stderr,
) -> None:
    ID_KEY = "id"
    ERROR_KEY = "error"
    STATUS_KEY = "status"
    PROG_KEY = "progress"
    LINE_CLEAR = "\033[2K"
    BLUE = "\033[34m"
    ENDC = "\033[0m"
    HEADER = f"{BLUE}docker push {ENDC}"

    def lines_up(lines: int) -> str:
        return f"\033[{lines}F"

    def lines_down(lines: int) -> str:
        return f"\033[{lines}E"

    ids = []
    for event in events:
        if ERROR_KEY in event:
            raise RuntimeError(f"failed to push docker image: {event[ERROR_KEY]}")

        id = event.get(ID_KEY)
        status = event.get(STATUS_KEY)

        if not status:
            continue

        if id:
            msg = f"{HEADER}{id}: {status} {event.get(PROG_KEY, '')}"
            if id not in ids:
                ids.append(id)
                stream.write(f"{msg}\n")
            else:
                lineno = len(ids) - ids.index(id)
                stream.write(f"{lines_up(lineno)}{LINE_CLEAR}{msg}{lines_down(lineno)}")
        else:
            stream.write(f"{HEADER}{status}\n")

def _args_join(args: Iterable[str]) -> str:
    """
    _args_join is like shlex.join but if the argument is wrapped in _noquote
    it'll not quote that argument.
    """
    quoted = [arg if isinstance(arg, _noquote) else shlex.quote(arg) for arg in args]
    return " ".join(quoted)