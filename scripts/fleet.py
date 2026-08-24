#!/usr/bin/env python3
"""Run attic-gensfen across a set of machines.

    fleet.py launch [--nodes A,B] [gensfen args...]
    fleet.py status [--nodes A,B]
    fleet.py watch  [--nodes A,B] [--interval 10]
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
import time
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

# Matched with `pgrep -x` against the executable name, never `-f` against the
# whole cmdline. Over SSH the remote shell carries the entire launch command,
# binary name included, so a cmdline match counts that shell as a second process
# — and `pkill -f` would kill the connection issuing it.
PATTERN = "attic-gensfen"

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, RED, CYAN, GREY = (
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[36m",
    "\033[90m",
)


def paint(text: str, colour: str, tty: bool) -> str:
    return f"{colour}{text}{RESET}" if tty else text


def bar(fraction: float, width: int, tty: bool) -> str:
    """A meter, coloured by how full it is.

    Green while there is room, yellow as it fills, red at the ceiling — the
    same reading as a utilisation gauge, where a pegged bar is the thing worth
    noticing.
    """
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    colour = GREEN if fraction < 0.7 else YELLOW if fraction < 0.95 else RED
    return paint("█" * filled, colour, tty) + paint("░" * (width - filled), GREY, tty)


def human(n: int) -> str:
    """Byte count, switching units before the figure reaches four digits.

    The switch is at 1000 of a unit rather than 1024, so 1010 MB reads as
    0.99 GB instead: a four-digit MB figure is harder to compare at a glance
    than a fractional GB one, which is the whole point of scaling it.
    """
    for unit, scale in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20)):
        if n >= scale * 1000 // 1024:
            return f"{n / scale:.1f} {unit}"
    return f"{n / 1024:.0f} KB"


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
    if on(node, f"pgrep -x '{PATTERN}' >/dev/null 2>&1")[0] == 0:
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
    if on(node, f"sleep 2; pgrep -x '{PATTERN}' >/dev/null 2>&1")[0] == 0:
        return f"{node}: running"
    _, tail = on(node, f"tail -1 {opts.out}/$(hostname -s).log 2>/dev/null")
    return f"{node}: did not start — {tail or 'no output'}"


class Node:
    """One machine's line in the status table."""

    def __init__(
        self,
        name: str,
        procs: int,
        conc: str,
        games: int,
        written: int,
        uptime: int,
        record_size: int | None,
    ):
        self.name = name
        self.procs = procs
        self.conc = conc
        self.games = games
        self.written = written
        self.uptime = uptime
        self.record_size = record_size

    @property
    def positions(self) -> int | None:
        """Positions written, from the byte count and the record size.

        Exact for the fixed-width formats and unknowable for JSONL, where a
        record's length depends on the position in it.
        """
        if not self.record_size:
            return None
        return self.written // self.record_size

    @property
    def lifetime_rate(self) -> float | None:
        """Games per second since the process started.

        A single sample has nothing to difference against, so this is what one
        `status` call can report. `watch` prefers the rate since its previous
        frame, which tracks a throttled node instead of averaging over the time
        before it was throttled.
        """
        return self.games / self.uptime if self.uptime > 0 else None

    @property
    def alive(self) -> bool:
        return self.procs > 0


def probe(node: str, opts: argparse.Namespace) -> Node:
    """Processes, concurrency, games, and bytes written on `node`."""
    _, out = on(
        node,
        # `pgrep -c` prints 0 *and* exits non-zero when nothing matches, so a
        # `|| echo 0` fallback would append a second line.
        f"log={opts.out}/$(hostname -s).log; "
        f"p=$(pgrep -c -x '{PATTERN}' 2>/dev/null | head -1); "
        # The governor only logs when it changes something, so an untouched run
        # has no concurrency line at all — fall back to the ceiling it started
        # from rather than reporting zero.
        f"c=$(grep -h concurrency= $log 2>/dev/null | tail -1 "
        f"   | sed 's/.*concurrency=\\([0-9]*\\).*/\\1/'); "
        f"[ -n \"$c\" ] || c=$(grep -ho 'ceiling=[0-9]*' $log 2>/dev/null | tail -1 | cut -d= -f2); "
        f"[ -n \"$c\" ] || c=$(pgrep -a -x '{PATTERN}' 2>/dev/null | head -1 "
        f"   | grep -o -- '--threads [0-9]*' | cut -d' ' -f2); "
        # Every worker reports the process-wide total, so the last line of any
        # of them is the count — no need to sum per-worker figures.
        f"g=$(grep -ho 'fleet [0-9]*' $log 2>/dev/null | tail -1 | cut -d' ' -f2); "
        f"b=$(du -cb {opts.out}/$(hostname -s)*.bin {opts.out}/$(hostname -s)*.jsonl "
        f"   2>/dev/null | tail -1 | cut -f1); "
        f"u=$(ps -o etimes= -p $(pgrep -x '{PATTERN}' 2>/dev/null | head -1) 2>/dev/null "
        f"   | tr -d ' '); "
        # The fixed-width formats have no header, so the record size has to come
        # from the arguments the run was started with.
        f"a=$(pgrep -a -x '{PATTERN}' 2>/dev/null | head -1); "
        f"fmt=$(echo \"$a\" | grep -o -- '--format [a-z]*' | cut -d' ' -f2); "
        f"mpv=$(echo \"$a\" | grep -o -- '--multipv [0-9]*' | cut -d' ' -f2); "
        f'printf "%s %s %s %s %s %s %s" "${{p:-0}}" "${{c:-0}}" "${{g:-0}}" '
        f'"${{b:-0}}" "${{u:-0}}" "${{fmt:-jsonl}}" "${{mpv:-5}}"',
    )
    f = (out.split() + ["0"] * 7)[:7]
    num = lambda s: int(s) if s.isdigit() else 0
    return Node(
        node,
        num(f[0]),
        f[1] if f[1].isdigit() else "-",
        num(f[2]),
        num(f[3]),
        num(f[4]),
        record_bytes(f[5], num(f[6])),
    )


def record_bytes(fmt: str, multipv: int) -> int | None:
    """Bytes per position, or `None` where records are not fixed width.

    `psv` is upstream's 40-byte record; `pack` appends `multipv` pairs of
    (move16, centipawn). JSONL varies per position, so a byte count says
    nothing about how many are in it.
    """
    if fmt == "psv":
        return 40
    if fmt == "pack":
        return 40 + 4 * max(multipv, 0)
    return None


def stop(node: str, opts: argparse.Namespace) -> str:
    del opts
    _, left = on(
        node,
        f"pkill -x '{PATTERN}' >/dev/null 2>&1; sleep 2; "
        f"pgrep -c -x '{PATTERN}' 2>/dev/null | head -1",
    )
    return f"{node:<5} remaining={left or 0}"


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True, description=__doc__)
    parser.add_argument("command", choices=["launch", "status", "stop", "watch"])
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="seconds between refreshes in watch mode",
    )
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
        elif opts.command == "status":
            render(list(pool.map(lambda n: probe(n, opts), nodes)), None)
        else:
            watch(pool, nodes, opts)
    return 0


def watch(pool: ThreadPoolExecutor, nodes: list[str], opts: argparse.Namespace) -> None:
    """Redraw until interrupted, showing the rate since the previous sample."""
    previous: tuple[float, dict[str, int]] | None = None
    try:
        while True:
            sampled = time.monotonic()
            rows = list(pool.map(lambda n: probe(n, opts), nodes))

            rate = None
            if previous is not None:
                before, counts = previous
                elapsed = sampled - before
                if elapsed > 0:
                    rate = {
                        r.name: max(0.0, (r.games - counts.get(r.name, r.games)) / elapsed)
                        for r in rows
                    }
            previous = (sampled, {r.name: r.games for r in rows})

            # Home the cursor and clear forward rather than clearing first: a
            # full wipe makes the table blink on every refresh.
            sys.stdout.write("\033[H\033[J" if sys.stdout.isatty() else "\n")
            render(rows, rate)
            sys.stdout.flush()
            time.sleep(opts.interval)
    except KeyboardInterrupt:
        print()


def render(rows: list[Node], rate: dict[str, float] | None) -> None:
    """One frame of the table. `rate` is games/sec per node, when known."""
    tty = sys.stdout.isatty()
    # The bar is read against the busiest machine rather than an absolute
    # ceiling: what matters here is which nodes are lagging the fleet, and
    # concurrency ceilings differ between machines anyway.
    peak = max((r.games for r in rows), default=0) or 1

    head = (
        f"{'NODE':<6} {'PROC':>4} {'CONC':>5} {'GAMES':>9} {'GAMES/H':>9} "
        f"{'POSITIONS':>12}  {'PROGRESS':<22} {'WRITTEN':>9}"
    )
    print(paint(head, BOLD, tty))

    for r in rows:
        if not r.alive:
            print(
                paint(
                    f"{r.name:<6} {'—':>4} {'—':>5} {'—':>9} {'—':>9} {'—':>12}  "
                    f"{'not running':<22} {'—':>9}",
                    GREY,
                    tty,
                )
            )
            continue
        per_second = (rate or {}).get(r.name) or r.lifetime_rate
        speed = f"{per_second * 3600:,.0f}" if per_second else "—"
        positions = f"{r.positions:,}" if r.positions is not None else "—"
        print(
            f"{paint(f'{r.name:<6}', CYAN, tty)} {r.procs:>4} {r.conc:>5} "
            f"{r.games:>9,} {speed:>9} {positions:>12}  "
            f"{bar(r.games / peak, 20, tty)}  {human(r.written):>9}"
        )

    live = [r for r in rows if r.alive]
    games = sum(r.games for r in rows)
    written = sum(r.written for r in rows)
    counted = [r.positions for r in rows if r.positions is not None]
    summary = f"{len(live)}/{len(rows)} nodes   {games:,} games"
    if counted:
        summary += f"   {sum(counted):,} positions"
    summary += f"   {human(written)}"
    total_rate = sum(
        (rate or {}).get(r.name) or r.lifetime_rate or 0.0 for r in rows
    )
    if total_rate:
        summary += f"   {total_rate * 3600:,.0f} games/h"
    print()
    print(paint(summary, BOLD, tty))
    if tty:
        print(paint(f"{time.strftime('%H:%M:%S')}  ctrl-c to exit", DIM, tty))


if __name__ == "__main__":
    sys.exit(main())
