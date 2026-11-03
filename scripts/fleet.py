#!/usr/bin/env python3
"""Run attic-gensfen across a set of machines.

    fleet.py launch [--nodes A,B] [--block N] [gensfen args...]
    fleet.py status [--nodes A,B] [--no-scan]
    fleet.py watch  [--nodes A,B] [--interval 10] [--no-scan]
    fleet.py stop   [--nodes A,B]

`status` and `watch` report two counts of games, because there are two. `RUN` is
what the running process has produced; `TOTAL` is what the shards on disk hold,
which includes any run that wrote to them before. Only `TOTAL` divides into
`POSITIONS`. Counting the second one means reading the shards, which `--no-scan`
skips.

Unrecognised arguments are passed to attic-gensfen, so `--depth 9
--format pack` works without this script knowing what those mean. Two are read
here rather than passed on unchanged, because a per-process meaning is the
wrong one for a fleet: `--seed` is a base the blocks count up from, since the
same seed on two machines means the same random moves at the same plies, and
`--games` is the total for the whole fleet. Every other quota, `--positions`
included, is still per node.

A `--games` total is not divided into shares. The nodes take blocks from a
shared queue until the fleet has played the number asked for, so a machine
half as fast takes half as many blocks and the run ends on all of them at
once. `take_games` has the measurements that ruled out an even share and
`block_size` what sets a block.

Run from wherever you have SSH to the nodes; they are not assumed to reach each
other. The reference deployment has no agent forwarding, so a node cannot start
its neighbours even though the filesystem is shared.
"""

from __future__ import annotations

import argparse
import base64
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

# Counts games in the shards themselves, run on the node that owns them.
#
# The process counter only knows about its own games, while the shards are
# opened for append and may already hold a previous run's. Dividing one by the
# other gives a plies-per-game figure that is wrong by however much came before.
#
# A game boundary is a ply that did not increase. Reading `ply == 1` would miss
# any game whose first record is not ply 1, and each shard belongs to exactly
# one worker, so the records in it are already in order.
SCAN_PY = """import glob, sys
import numpy as np

prefix, record = sys.argv[1], int(sys.argv[2])
games = positions = 0
for path in sorted(glob.glob(prefix + ".w*.bin")):
    raw = np.fromfile(path, dtype=np.uint8)
    n = raw.size // record
    if n == 0:
        continue
    ply = raw[: n * record].reshape(n, record)[:, 36:38].copy().view(np.uint16)
    ply = ply.ravel()
    games += 1 + int((ply[1:] <= ply[:-1]).sum())
    positions += n
print(games, positions)
"""

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


# SSH's own exit code when it cannot reach the machine, and what `on` reports
# when the call times out. Remote commands pass their status through unchanged,
# so nothing this script runs remotely returns it: 255 means the *connection*
# failed, not the command. Every caller that reads a status as "no" has to
# separate the two, because "the node said no" and "the node did not answer"
# lead to opposite decisions.
SSH_FAILED = 255


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
        return SSH_FAILED, ""
    return done.returncode, done.stdout.strip()


def wrap_detached(command: str, log: str) -> str:
    """A command that outlives the SSH session without holding it open."""
    return f"setsid sh -c {shlex.quote(command)} >{log} 2>&1 </dev/null & disown 2>/dev/null; exit 0"


def take_seed(passthrough: list[str]) -> tuple[int, list[str]]:
    """The base seed, taken out of the arguments so each node can get its own.

    A worker's seed is `process seed * 1000003 + worker index`, so one seed
    handed to the fleet is the same seed on every machine: eight nodes drawing
    the same random moves at the same plies. Measured, they do not currently
    write identical games — the workers on a node share a transposition table,
    and which of them reaches an entry first is a race — but that is a side
    effect of threading rather than anything the generator promises.

    With a `--games` total the count runs over blocks rather than nodes: block
    `n` is given `base + n`, wherever it is played. Without one, each node is
    given `base + its position in --nodes`. Either way a process holds a few
    hundred workers at most, far short of the 1000003 stride, so no two of
    their ranges can overlap.

    Without `--seed` the base follows the clock. Output is opened for append,
    so a fleet stopped and started on the same seed would write a second copy
    of the games already on disk; moving the base means a restart continues the
    corpus instead of repeating it. Pass `--seed` to reproduce a run — either
    way, the value each node was given is printed.
    """
    if "--seed" not in passthrough:
        return int(time.time()), list(passthrough)
    at = passthrough.index("--seed")
    if at + 1 == len(passthrough):
        raise SystemExit("--seed needs a value")
    try:
        base = int(passthrough[at + 1])
    except ValueError:
        raise SystemExit(f"--seed: cannot parse {passthrough[at + 1]!r}") from None
    return base, passthrough[:at] + passthrough[at + 2 :]


def take_games(passthrough: list[str]) -> tuple[int, list[str]]:
    """The fleet's total game count, taken out of the arguments.

    attic-gensfen counts games per process and one process runs per node, so a
    fleet handed `--games 100000` plays 100000 on each machine — eight times
    the corpus, over eight times the days. A quota is the one number a run is
    planned around, so being wrong about it by the size of the fleet is not a
    detail to leave to whoever reads the table afterwards.

    It comes back as a total rather than as a share each, because an even share
    is only right when the machines are equally fast and they are not. On
    2026-09-07 the same 12,500 games took 22m17s on g19 and 40m20s on g24; the
    run lasted 40m20s and for the last 18 minutes of it g24 was the only node
    still playing. Free cores do not say which node that will be either — two
    minutes in, g24 and g25 were both admitting 252 workers, and g24 had
    produced 248 games to g25's 557. Speed is only visible once a machine is
    running, so the work is handed out in blocks and the nodes take what they
    can carry.

    Zero is the generator's own way of saying "until killed". It is handed back
    as no total at all, which leaves one unbounded process per node and no
    queue: what this script did before blocks existed.

    Only `--games` is treated this way. `--positions` counts per process too,
    but a block is a process, so a positions quota would be paid again on every
    block; `main` refuses the pair rather than quietly multiplying one of them.
    """
    if "--games" not in passthrough:
        return 0, list(passthrough)
    at = passthrough.index("--games")
    if at + 1 == len(passthrough):
        raise SystemExit("--games needs a value")
    try:
        total = int(passthrough[at + 1])
    except ValueError:
        raise SystemExit(f"--games: cannot parse {passthrough[at + 1]!r}") from None
    if total < 0:
        raise SystemExit(f"--games {total} is not a count")
    return total, passthrough[:at] + passthrough[at + 2 :]


def block_size(total: int, nodes: int) -> int:
    """How many games one block is, when `--block` does not say.

    Two costs are paid once per block and neither shrinks with it. Starting the
    process costs 19 seconds before its first game — loading the evaluation
    function and reading the start positions; measured on g19 as 19.08 s at one
    thread and 19.29 s at eight, so it is not thread setup and cutting threads
    will not avoid it. And the quota is checked between games, never inside
    one, so every worker already playing when it trips finishes what it is
    doing: 115 to 241 games past the number asked for, measured across the
    eight 256-core nodes on 2026-09-07.

    Against those, the longest tail a run can end with is one block on the
    slowest machine. Five blocks a node is where that balance sits at the sizes
    this fleet is used at — for 100,000 games over eight nodes, blocks of 2,500
    and a tail of at most a few minutes. The floor keeps a small run from being
    cut so fine that a block is no bigger than the overshoot on it.
    """
    return max(1000, total // (5 * nodes))


def queue_path(opts: argparse.Namespace) -> str:
    """Where blocks are claimed. On the shared home, so every node sees it.

    Named after the output directory rather than fixed, so two corpora being
    built at once do not hand each other's blocks out. Only the claims cross
    NFS — a few bytes per block — while the shards stay node-local.
    """
    name = opts.out.rstrip("/").rsplit("/", 1)[-1]
    if not name or name in (".", ".."):
        raise SystemExit(f"--out {opts.out!r} has no name to build a queue from")
    return f"$HOME/.attic-fleet/{name}"


def reset_queue(node: str, queue: str) -> None:
    """Empty the queue for a new launch.

    Cleared rather than continued: the claims are what tells a dispatcher the
    run is done, so starting against a finished queue would hand out nothing at
    all, and starting against a half-finished one would replay its blocks into
    shards that are opened for append.
    """
    code, out = on(node, f"rm -rf {queue} && mkdir -p {queue}/claim {queue}/done && echo ok")
    if out != "ok":
        raise SystemExit(f"could not prepare the queue on {node}: {out or code}")


def dispatch(run: str, opts: argparse.Namespace, plan: tuple[str, int, int, int]) -> str:
    """The shell a node loops in: claim a block, play it, report what it made.

    A claim is `mkdir` on the shared queue, which is atomic on NFS — two nodes
    reaching for the same number, one of them creates the directory and the
    other moves on to the next. Nothing else coordinates them.

    The block is claimed before it is sized, not after. Sized first, eight nodes
    reading the queue in the same moment all see the same room and all take it:
    on a 10,000-game rehearsal with instant blocks that ran 7% long. Claiming
    first makes the number itself the reservation — every node counts the
    lower-numbered claims that have not reported yet at a full block each, so a
    node holding a higher number always sizes itself smaller than the one below
    it and the same games cannot be handed out twice.

    Blocks report the games they actually played rather than the ones they were
    asked for, which is what holds the overshoot to one round instead of one per
    block. Each block runs a little past its own quota — the generator checks it
    between games, never inside one — and counting the requested figure would
    let that error collect on every restart. The fleet lands slightly over the
    number asked for, as a single process always did.

    A block that adds no `finished` line to the log produced nothing — it was
    killed, or it died on its arguments — so the node reports zero and stops
    rather than spinning through the queue at 19 seconds a turn. Whether the
    marker is still there separates the two: gone means `stop` took it, and the
    log should say so rather than blame the block for a kill that was asked for.

    The marker file is the loop's permission to continue. `stop` removes it so
    a node between blocks does not start another, and `status` reads it to tell
    a machine loading its next block from one that has finished.
    """
    queue, total, block, base = plan
    return (
        f"q={queue}; m={opts.out}/$(hostname -s).loop; log={opts.out}/$(hostname -s).log; "
        f'mkdir -p "$q/claim" "$q/done"; : > "$m"; trap \'rm -f "$m"\' EXIT INT TERM; '
        # `status` reads the record size off the running process, and between
        # blocks there is not one. Writing the command down once leaves it
        # somewhere to look.
        f'echo "fleet: run {run}"; '
        "seq=0; "
        'while [ -e "$m" ]; do '
        '  while [ -d "$q/claim/$seq" ]; do seq=$((seq+1)); done; '
        '  mkdir "$q/claim/$seq" 2>/dev/null || { seq=$((seq+1)); continue; }; '
        # Both figures come from one listing. Read separately, a report landing
        # between the two scans is counted in neither, and the block sizes
        # itself as though those games were still to play.
        '  made=0; seen=" "; '
        '  for f in "$q"/done/*; do '
        '    [ -f "$f" ] || continue; '
        '    v=$(cat "$f"); made=$(( made + ${v:-0} )); seen="$seen${f##*/} "; '
        '  done; '
        '  ahead=0; i=0; '
        '  while [ "$i" -lt "$seq" ]; do '
        '    case "$seen" in *" $i "*) ;; *) ahead=$(( ahead + 1 ));; esac; '
        '    i=$(( i + 1 )); '
        '  done; '
        f'  want=$(( {total} - made - ahead * {block} )); '
        f'  [ "$want" -gt {block} ] && want={block}; '
        '  if [ "$want" -le 0 ]; then '
        '    echo 0 > "$q/done/$seq"; echo "fleet: nothing left to claim"; break; '
        '  fi; '
        f'  echo "fleet: block $seq, $want games, seed $(( {base} + seq ))"; '
        '  before=$(grep -c "gensfen: finished," "$log" 2>/dev/null); '
        f'  {run} --games "$want" --seed $(( {base} + seq )); '
        '  after=$(grep -c "gensfen: finished," "$log" 2>/dev/null); '
        '  got=0; [ "${after:-0}" -gt "${before:-0}" ] && '
        '    got=$(grep "gensfen: finished," "$log" | tail -1 | awk \'{print $3}\'); '
        '  echo "${got:-0}" > "$q/done/$seq"; '
        '  if [ "${got:-0}" -eq 0 ]; then '
        '    if [ -e "$m" ]; then echo "fleet: block $seq produced nothing, stopping"; '
        '    else echo "fleet: stopped during block $seq"; fi; '
        '    break; '
        '  fi; '
        "  seq=$((seq+1)); "
        "done; "
        'rm -f "$m"; echo "fleet: done"'
    )


def running(node: str, opts: argparse.Namespace) -> bool | None:
    """Whether `node` is busy: playing a block, or between two of them.

    `None` means the question could not be asked. Returning `False` there would
    be the dangerous answer: the caller clears the shared queue on the strength
    of this, and a node that is merely unreachable is very likely still playing.
    Eight nodes at once can go unreachable over a cold jump host without one of
    them having stopped.

    The marker has to be checked as well as the process. There are 19 seconds
    between one block ending and the next reaching its first game, and a node
    caught in that window looks idle to `pgrep` while still holding a place in
    the queue.
    """
    status, _ = on(node, f"pgrep -x '{PATTERN}' >/dev/null 2>&1")
    if status == SSH_FAILED:
        return None
    if status == 0:
        return True
    status, _ = on(node, f"[ -e {opts.out}/$(hostname -s).loop ]")
    return None if status == SSH_FAILED else status == 0


def launch(
    node: str,
    seed: int,
    passthrough: list[str],
    opts: argparse.Namespace,
    plan: tuple[str, int, int, int] | None = None,
) -> str:
    """Start one node, or explain why not.

    Without a plan the node runs a single unbounded process, as it always did.
    With one it runs the dispatcher, which starts a process per block and gets
    its seed from the block number instead of from the node's position in
    `--nodes`.
    """
    # Refuse rather than stack a second run on the first: two of them on one
    # machine halve each other and interleave into the same files. A node that
    # did not answer is refused on the same grounds — it cannot be shown to be
    # idle, and the cost of being wrong is a corrupted output file.
    state = running(node, opts)
    if state is None:
        return f"{node}: unreachable — not started"
    if state:
        return f"{node}: already running"

    args = " ".join(shlex.quote(a) for a in passthrough)
    # `--threads` is a ceiling here; the process adjusts downward on its own as
    # the machine fills, so this only has to be a sane starting point.
    run = (
        f"nice -n {opts.nice} {opts.bin} --eval-dir {opts.eval_dir} "
        f"--out {opts.out}/$(hostname -s) {args}"
    )
    command = f"{run} --seed {seed}" if plan is None else dispatch(run, opts, plan)
    on(node, f"mkdir -p {opts.out} && " + wrap_detached(command, f"{opts.out}/$(hostname -s).log"))

    # Confirm by looking for the process. A detached launch reports success even
    # when the binary rejects its arguments and exits immediately.
    if on(node, f"sleep 2; pgrep -x '{PATTERN}' >/dev/null 2>&1")[0] == 0:
        if plan is None:
            return f"{node}: running (seed {seed})"
        return f"{node}: running (blocks of {plan[2]:,}, seed base {plan[3]})"
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
        corpus_games: int | None = None,
        corpus_positions: int | None = None,
        looping: bool = False,
        reachable: bool = True,
    ):
        self.name = name
        self.procs = procs
        self.conc = conc
        self.games = games
        self.written = written
        self.uptime = uptime
        self.record_size = record_size
        self.corpus_games = corpus_games
        self.corpus_positions = corpus_positions
        self.looping = looping
        self.reachable = reachable

    @property
    def positions(self) -> int | None:
        """Positions on disk: counted where the shards were scanned.

        The byte estimate is the fallback. It is exact for the fixed-width
        formats and unknowable for JSONL, where a record's length depends on
        the position in it.
        """
        if self.corpus_positions is not None:
            return self.corpus_positions
        if not self.record_size:
            return None
        return self.written // self.record_size

    @property
    def plies(self) -> float | None:
        """Plies per game, from figures that cover the same games.

        `games` counts this process's output and the shards may hold earlier
        runs too, so the two must not be divided into each other.
        """
        if not self.corpus_games or self.corpus_positions is None:
            return None
        return self.corpus_positions / self.corpus_games

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
        """Working, whether or not a process is on the machine this second.

        A node between blocks has none for the 19 seconds its next one spends
        loading. Reading that as "not running" would blank the row, and worse,
        would make `launch` willing to start a second dispatcher on top.

        An unreachable node is not alive here — nothing can be reported for it —
        but the row says which of the two it is, so that a fleet that has gone
        quiet is not mistaken for a fleet that has finished.
        """
        return self.reachable and (self.procs > 0 or self.looping)


def probe(node: str, opts: argparse.Namespace) -> Node:
    """Processes, concurrency, games, and bytes written on `node`.

    A failed SSH parses into the same zeroes a genuinely idle node reports, so
    the status is kept and carried on the row. Without it a run whose nodes
    have merely gone unreachable reads as a run that has finished, and the
    fleet totals quietly lose whatever those nodes had played.
    """
    status, out = on(
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
        # Every worker reports the process-wide total, so the last of their
        # lines is the count for the block it is in — but a node plays several
        # blocks into one log, and each starts its counter again. The finished
        # blocks are summed and the live one added on top; `c=0` after each
        # `finished` line is what keeps the last block's tail from being
        # counted twice once its own total has landed.
        "g=$(awk '/gensfen: finished,/ {s+=$3; c=0; next} "
        "/\\(fleet [0-9]+\\)$/ {t=$NF; gsub(/[()]/, \"\", t); c=t+0} "
        "END {print s+c+0}' $log 2>/dev/null); "
        f"b=$(du -cb {opts.out}/$(hostname -s)*.bin {opts.out}/$(hostname -s)*.jsonl "
        f"   2>/dev/null | tail -1 | cut -f1); "
        # The marker is made once, at the start of the run, so it dates the
        # whole of it. `ps` would date the current block, and a rate worked out
        # from that would climb through every block and reset at each boundary.
        f"m={opts.out}/$(hostname -s).loop; l=0; "
        f'[ -e "$m" ] && l=1; '
        f'u=$(if [ -e "$m" ]; then expr $(date +%s) - $(stat -c %Y "$m"); '
        f"   else ps -o etimes= -p $(pgrep -x '{PATTERN}' 2>/dev/null | head -1) 2>/dev/null "
        f"        | tr -d ' '; fi); "
        # The fixed-width formats have no header, so the record size has to come
        # from the arguments the run was started with.
        f"a=$(pgrep -a -x '{PATTERN}' 2>/dev/null | head -1); "
        f"fmt=$(echo \"$a\" | grep -o -- '--format [a-z]*' | cut -d' ' -f2); "
        f"mpv=$(echo \"$a\" | grep -o -- '--multipv [0-9]*' | cut -d' ' -f2); "
        f"a2=$(grep -m1 '^fleet: run ' $log 2>/dev/null); "
        f"[ -n \"$fmt\" ] || fmt=$(echo \"$a2\" | grep -o -- '--format [a-z]*' | cut -d' ' -f2); "
        f"[ -n \"$mpv\" ] || mpv=$(echo \"$a2\" | grep -o -- '--multipv [0-9]*' | cut -d' ' -f2); "
        + scan_command(opts)
        + 'printf "%s %s %s %s %s %s %s %s %s" "${p:-0}" "${c:-0}" "${g:-0}" '
        '"${b:-0}" "${u:-0}" "${fmt:-jsonl}" "${mpv:-5}" "${s:-0/0}" "${l:-0}"',
    )
    f = (out.split() + ["0"] * 9)[:9]
    num = lambda s: int(s) if s.isdigit() else 0
    # A scan that did not happen reports zeroes, and a zero here has to stay
    # distinct from a counted zero: it is what makes `positions` fall back to
    # the byte estimate rather than claiming the shards are empty.
    scanned = (f[7].split("/") + ["0"])[:2]
    return Node(
        node,
        num(f[0]),
        f[1] if f[1].isdigit() else "-",
        num(f[2]),
        num(f[3]),
        num(f[4]),
        record_bytes(f[5], num(f[6])),
        num(scanned[0]) or None,
        num(scanned[1]) or None,
        f[8] == "1",
        status != SSH_FAILED,
    )


def scan_command(opts: argparse.Namespace) -> str:
    """Shell that counts games in this node's shards, or nothing if disabled.

    The script travels base64-encoded. It is Python inside a shell command
    inside an SSH argument, and every layer between here and there has its own
    opinion about quotes.
    """
    if not opts.scan:
        return ""
    payload = base64.b64encode(SCAN_PY.encode()).decode()
    return (
        # `psv` has no candidate list appended; `jsonl` is not fixed width, so
        # there is nothing to stride through and it is left to the byte estimate.
        f'rec=$(( 40 + 4 * ${{mpv:-5}} )); [ "${{fmt:-jsonl}}" = psv ] && rec=40; '
        f's=""; case "${{fmt:-jsonl}}" in pack|psv) '
        f's=$(echo {payload} | base64 -d '
        f'| nice -n {opts.nice} python3 - "{opts.out}/$(hostname -s)" "$rec" 2>/dev/null '
        f'| tr " " "/");; esac; '
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
    """Stop the node: the dispatcher's permission first, then the process.

    The marker goes before the kill, and not after. A node killed while its
    loop still holds permission claims the next block and starts again, which
    is the opposite of stopping; taking the marker first means the loop finds
    it gone the moment the process it is waiting on dies.

    Only `--out` is needed for this. The queue lives elsewhere and is not
    touched here, so stopping never depends on working out where it is.

    It waits for the process to be gone rather than sleeping a fixed time and
    reporting whatever it finds. A process with 250 workers in flight takes a
    few seconds to unwind, and a count taken too early reads as "still running"
    on a node that is halfway through exiting — which invites a second kill on
    something that was already stopping.
    """
    _, left = on(
        node,
        f"rm -f {opts.out}/$(hostname -s).loop; "
        f"pkill -x '{PATTERN}' >/dev/null 2>&1; "
        "i=0; while [ $i -lt 30 ]; do "
        f"  pgrep -x '{PATTERN}' >/dev/null 2>&1 || break; "
        "  sleep 1; i=$((i+1)); "
        "done; "
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
    parser.add_argument(
        "--block",
        type=int,
        default=0,
        help="games per block; default is a fifth of the total per node",
    )
    parser.add_argument(
        "--no-scan",
        dest="scan",
        action="store_false",
        help="do not read the shards; report positions from the byte count",
    )
    opts, passthrough = parser.parse_known_args()

    nodes = [n for n in opts.nodes.split(",") if n]
    # Fanned out because a launch waits two seconds on each node to confirm, and
    # an unreachable one burns the full connect timeout.
    with ThreadPoolExecutor(max_workers=len(nodes) or 1) as pool:
        if opts.command == "launch":
            base, passthrough = take_seed(passthrough)
            total, passthrough = take_games(passthrough)
            plan = None
            if total:
                if "--positions" in passthrough:
                    raise SystemExit(
                        "--positions counts per process, and a block is a process: "
                        "it would be paid again on every block. Use one quota."
                    )
                # Checked before the queue is cleared rather than inside the
                # launch, because clearing it under a fleet that is still
                # running would set every node counting from nothing.
                # Materialised: `pool.map` hands back a one-shot iterator and
                # both checks below walk it.
                live = list(pool.map(lambda n: running(n, opts), nodes))
                lost = [n for n, b in zip(nodes, live, strict=True) if b is None]
                if lost:
                    raise SystemExit(
                        f"could not reach {', '.join(lost)} — the queue was left alone. "
                        "A node that does not answer has not been shown to be idle, "
                        "and the check exists to keep the queue from being cleared "
                        "under a fleet that is still playing."
                    )
                busy = [n for n, b in zip(nodes, live, strict=True) if b]
                if busy:
                    raise SystemExit(
                        f"already running on {', '.join(busy)} — the queue was left alone"
                    )
                block = min(opts.block or block_size(total, len(nodes)), total)
                plan = (queue_path(opts), total, block, base)
                reset_queue(nodes[0], plan[0])
                print(f"{total:,} games in blocks of {block:,}, claimed from {plan[0]}")
            for line in pool.map(
                lambda pair: launch(pair[0], pair[1], passthrough, opts, plan),
                zip(nodes, range(base, base + len(nodes)), strict=True),
            ):
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
    peak = max((r.corpus_games or r.games for r in rows), default=0) or 1

    # `RUN` is what the live process has produced; `TOTAL` is what the shards
    # hold, earlier runs included. Naming both of them "games" is how a
    # plies-per-game figure of 196 once got reported.
    head = (
        f"{'NODE':<6} {'PROC':>4} {'CONC':>5} {'RUN':>9} {'GAMES/H':>9} "
        f"{'TOTAL':>9} {'POSITIONS':>12} {'PLY/G':>6}  {'PROGRESS':<22} {'WRITTEN':>9}"
    )
    print(paint(head, BOLD, tty))

    for r in rows:
        if not r.alive:
            state = "not running" if r.reachable else "unreachable"
            print(
                paint(
                    f"{r.name:<6} {'—':>4} {'—':>5} {'—':>9} {'—':>9} {'—':>9} "
                    f"{'—':>12} {'—':>6}  {state:<22} {'—':>9}",
                    GREY,
                    tty,
                )
            )
            continue
        # A node between blocks has no process to count. Blanking the column
        # would read as a dead machine, which is the one thing it is not.
        procs = f"{r.procs:>4}" if r.procs else f"{'···':>4}"
        per_second = (rate or {}).get(r.name) or r.lifetime_rate
        speed = f"{per_second * 3600:,.0f}" if per_second else "—"
        positions = f"{r.positions:,}" if r.positions is not None else "—"
        total = f"{r.corpus_games:,}" if r.corpus_games is not None else "—"
        plies = f"{r.plies:.1f}" if r.plies is not None else "—"
        print(
            f"{paint(f'{r.name:<6}', CYAN, tty)} {procs} {r.conc:>5} "
            f"{r.games:>9,} {speed:>9} {total:>9} {positions:>12} {plies:>6}  "
            f"{bar((r.corpus_games or r.games) / peak, 20, tty)}  {human(r.written):>9}"
        )

    live = [r for r in rows if r.alive]
    games = sum(r.games for r in rows)
    written = sum(r.written for r in rows)
    counted = [r.positions for r in rows if r.positions is not None]
    scanned = [r.corpus_games for r in rows if r.corpus_games is not None]
    summary = f"{len(live)}/{len(rows)} nodes   {games:,} games this run"
    if scanned:
        summary += f"   {sum(scanned):,} on disk"
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
    # Every figure above is a sum over the nodes that answered. Saying so is
    # the difference between a fleet that has finished and one that has only
    # stopped being visible from here.
    unreachable = [r.name for r in rows if not r.reachable]
    if unreachable:
        print(
            paint(
                f"{', '.join(unreachable)} did not answer — "
                "the totals cover the rest, and say nothing about these",
                GREY,
                tty,
            )
        )
    if tty:
        print(paint(f"{time.strftime('%H:%M:%S')}  ctrl-c to exit", DIM, tty))


if __name__ == "__main__":
    sys.exit(main())
