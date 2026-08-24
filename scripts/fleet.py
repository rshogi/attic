#!/usr/bin/env python3
"""Run attic-gensfen across a set of machines.

    fleet.py launch [--nodes A,B] [gensfen args...]
    fleet.py status [--nodes A,B]
    fleet.py stop   [--nodes A,B]

Unrecognised arguments are passed to attic-gensfen, so `--depth 9
--format pack` works without this script knowing what those mean.

Run from wherever you have SSH to the nodes; they are not assumed to reach each
other. The reference deployment has no agent forwarding, so a node cannot start
its neighbours even though the filesystem is shared.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# Binary and weights live on the shared home; node-local paths are visible from
# exactly one machine. Output stays node-local on purpose — every worker writes
# continuously and that traffic has no business crossing NFS.
DEFAULTS = {
    "nodes": "g17,g18,g19,g20,g21,g23,g24,g25",
    "bin": "$HOME/attic/bin/attic-gensfen",
    "eval_dir": "$HOME/izanami/assets/eval/halfka2304/nagisa_v4",
    "out": "/raid/work/smorimoto/gensfen",
}

# `[a]ttic` so the pattern cannot match the shell running it: over SSH the
# remote shell carries the whole command in its own cmdline, so a plain
# "attic-gensfen" makes pgrep find itself and pkill kill the connection.
PATTERN = "[a]ttic-gensfen"


def on(node: str, script: str, timeout: int = 60) -> tuple[int, str]:
    """Run `script` on `node`, returning (exit status, stdout).

    SSH waits for the remote *pipes* to close, not for the remote command to
    exit, so anything left holding stdout keeps this call open. Backgrounded
    work must therefore redirect all three streams; `wrap_detached` does.
    """
    try:
        done = subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 255, ""
    return done.returncode, done.stdout.strip()


def wrap_detached(command: str, log: str) -> str:
    """A command that outlives the SSH session without holding it open."""
    return f"setsid sh -c {shlex.quote(command)} >{log} 2>&1 </dev/null & disown 2>/dev/null; exit 0"


def launch(node: str, opts: argparse.Namespace, passthrough: list[str]) -> str:
    """Start one process on `node`, or explain why not."""
    # Refuse rather than stack a second process on the first: two runs on one
    # machine halve each other and interleave into the same files.
    if on(node, f"pgrep -f '{PATTERN}' >/dev/null 2>&1")[0] == 0:
        return f"{node}: already running"

    args = " ".join(shlex.quote(a) for a in passthrough)
    # `--threads` is a ceiling here; the process adjusts downward on its own as
    # the machine fills, so this only has to be a sane starting point.
    run = (
        f"nice -n {opts.nice} {opts.bin} --eval-dir {opts.eval_dir} "
        f"--out {opts.out}/$(hostname -s) {args}"
    )
    on(node, f"mkdir -p {opts.out} && " + wrap_detached(run, f"{opts.out}/$(hostname -s).log"))

    # Confirm by looking for the process. A detached launch reports success even
    # when the binary rejects its arguments and exits immediately.
    if on(node, f"sleep 2; pgrep -f '{PATTERN}' >/dev/null 2>&1")[0] == 0:
        return f"{node}: running"
    _, tail = on(node, f"tail -1 {opts.out}/$(hostname -s).log 2>/dev/null")
    return f"{node}: did not start — {tail or 'no output'}"


def status(node: str, opts: argparse.Namespace) -> tuple[str, int]:
    """Processes, concurrency, and bytes written on `node`."""
    _, out = on(
        node,
        # `pgrep -c` prints 0 *and* exits non-zero when nothing matches, so a
        # `|| echo 0` fallback would append a second line.
        f"p=$(pgrep -c -f '{PATTERN}' 2>/dev/null | head -1); "
        f"c=$(grep -h concurrency= {opts.out}/$(hostname -s).log 2>/dev/null "
        f"   | tail -1 | sed 's/.*concurrency=\\([0-9]*\\).*/\\1/'); "
        f"b=$(du -cb {opts.out}/$(hostname -s)* 2>/dev/null | tail -1 | cut -f1); "
        f'printf "%s %s %s" "${{p:-0}}" "${{c:-*}}" "${{b:-0}}"',
    )
    fields = out.split()
    procs, conc, raw = (fields + ["?", "*", "0"])[:3]
    written = int(raw) if raw.isdigit() else 0
    return f"{node:<5} proc={procs:<3} conc={conc:<5} {written // 1048576} MB", written


def stop(node: str, opts: argparse.Namespace) -> str:
    del opts
    _, left = on(
        node,
        f"pkill -f '{PATTERN}' >/dev/null 2>&1; sleep 2; "
        f"pgrep -c -f '{PATTERN}' 2>/dev/null | head -1",
    )
    return f"{node:<5} remaining={left or 0}"


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True, description=__doc__)
    parser.add_argument("command", choices=["launch", "status", "stop"])
    parser.add_argument("--nodes", default=DEFAULTS["nodes"], help="comma separated")
    parser.add_argument("--bin", default=DEFAULTS["bin"])
    parser.add_argument("--eval-dir", default=DEFAULTS["eval_dir"])
    parser.add_argument("--out", default=DEFAULTS["out"])
    parser.add_argument("--nice", type=int, default=19)
    opts, passthrough = parser.parse_known_args()

    nodes = [n for n in opts.nodes.split(",") if n]
    # Fanned out because a launch waits two seconds on each node to confirm, and
    # an unreachable one burns the full connect timeout.
    with ThreadPoolExecutor(max_workers=len(nodes) or 1) as pool:
        if opts.command == "launch":
            for line in pool.map(lambda n: launch(n, opts, passthrough), nodes):
                print(line)
        elif opts.command == "stop":
            for line in pool.map(lambda n: stop(n, opts), nodes):
                print(line)
        else:
            total = 0
            for line, written in pool.map(lambda n: status(n, opts), nodes):
                print(line)
                total += written
            print(f"\ntotal {total // 1048576} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
