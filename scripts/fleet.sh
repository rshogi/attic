#!/bin/sh
# Run attic-gensfen across a set of machines.
#
#   fleet.sh launch [args...]   start one process per node, passing args through
#   fleet.sh status             processes, load, and output size per node
#   fleet.sh stop               stop every node
#
# Nodes come from $NODES (default below). Everything else is passed to
# attic-gensfen unchanged, so `fleet.sh launch --depth 9 --format pack` works.
#
# Runs from wherever you have SSH to the nodes — they are not assumed to reach
# each other. The reference deployment has no agent forwarding, so a node cannot
# start its neighbours even though the filesystem is shared.
set -e

NODES=${NODES:-"g17 g18 g19 g20 g21 g23 g24 g25"}

# Binary and weights live on the shared home; `/raid` is node-local, so a build
# staged there is visible from exactly one machine. Output goes to `/raid` on
# purpose — every worker writes continuously, and that traffic has no business
# crossing NFS.
BIN=${BIN:-\$HOME/attic/bin/attic-gensfen}
EVAL_DIR=${EVAL_DIR:-\$HOME/izanami/assets/eval/halfka2304/nagisa_v4}
OUT=${OUT:-/raid/work/smorimoto/gensfen}

# Leave this many cores for whoever else is on the machine. These are shared
# hosts; `nice` yields the CPU but does not stop us from filling every core.
RESERVE=${RESERVE:-8}

# `[a]ttic` so the pattern does not match the shell running it. Over SSH the
# remote shell carries this whole script in its own cmdline, so a plain
# "attic-gensfen" makes pgrep find itself and pkill kill the connection.
PATTERN='[a]ttic-gensfen'

on() { timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }

# Physical cores, not logical. Past the physical count throughput regresses, so
# hyperthreads are not worth handing out.
CORES_CMD='lscpu -p=CORE 2>/dev/null | grep -v "^#" | sort -u | wc -l'

launch() {
  for node in $NODES; do
    info=$(on "$node" "echo \"\$($CORES_CMD) \$(cut -d' ' -f1 /proc/loadavg)\"" 2>/dev/null || echo "")
    if [ -z "$info" ]; then
      echo "$node: unreachable"
      continue
    fi

    cores=$(echo "$info" | cut -d' ' -f1)
    load=$(echo "$info" | cut -d' ' -f2 | cut -d. -f1)
    threads=$((cores - load - RESERVE))

    if [ "$threads" -lt 1 ]; then
      echo "$node: busy ($load of $cores cores), skipping"
      continue
    fi

    # Refuse rather than stack a second process on the first: two runs on one
    # node halve each other and the output interleaves into the same files.
    if on "$node" "pgrep -f '$PATTERN' >/dev/null 2>&1" 2>/dev/null; then
      echo "$node: already running, skipping"
      continue
    fi

    echo "$node: $threads threads ($load of $cores cores in use)"
    # Detached, so the exit status here says nothing about whether the run
    # survived — SSH reports on the shell that spawned it, not the process.
    # Confirm by looking for the process instead.
    on "$node" "mkdir -p $OUT && setsid nice -n 19 $BIN \
      --eval-dir $EVAL_DIR --out $OUT/\$(hostname -s) --threads $threads $* \
      >$OUT/\$(hostname -s).log 2>&1 < /dev/null &" >/dev/null 2>&1 || true

    sleep 2
    if on "$node" "pgrep -f '$PATTERN' >/dev/null 2>&1" 2>/dev/null; then
      echo "$node: running"
    else
      echo "$node: did not start — $(on "$node" "tail -1 $OUT/\$(hostname -s).log" 2>/dev/null)"
    fi
  done
}

status() {
  total=0
  for node in $NODES; do
    line=$(on "$node" "
      # \`pgrep -c\` prints 0 and *also* exits non-zero when nothing matches, so
      # a \`|| echo 0\` fallback would append a second line.
      p=\$(pgrep -c -f '$PATTERN' 2>/dev/null | head -1)
      l=\$(cut -d' ' -f1 /proc/loadavg)
      b=\$(du -cb $OUT/\$(hostname -s)* 2>/dev/null | tail -1 | cut -f1)
      printf '%s %s %s' \"\$p\" \"\$l\" \"\${b:-0}\"
    " 2>/dev/null || echo "? ? 0")

    procs=$(echo "$line" | cut -d' ' -f1)
    load=$(echo "$line" | cut -d' ' -f2)
    bytes=$(echo "$line" | cut -d' ' -f3)
    printf '%-5s proc=%-3s load=%-8s %s MB\n' \
      "$node" "$procs" "$load" "$((bytes / 1048576))"
    case "$bytes" in
      ''|*[!0-9]*) ;;
      *) total=$((total + bytes)) ;;
    esac
  done
  echo
  echo "total $((total / 1048576)) MB"
}

stop() {
  for node in $NODES; do
    left=$(on "$node" "
      pkill -f '$PATTERN' >/dev/null 2>&1
      sleep 2
      pgrep -c -f '$PATTERN' 2>/dev/null | head -1
    " 2>/dev/null || echo "?")
    printf '%-5s remaining=%s\n' "$node" "$left"
  done
}

case "${1:-}" in
  launch) shift; launch "$@" ;;
  status) status ;;
  stop)   stop ;;
  *) echo "usage: $0 launch [attic-gensfen args...] | status | stop" >&2; exit 2 ;;
esac
