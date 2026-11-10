#!/usr/bin/env python3
"""Put attic's evaluation of a Manaka network beside Manaka's own.

    manaka_compare.py --dump-eval PATH --attic-dump PATH --mnw FILE --nn FILE \
                      --sfens FILE [--limit N]

`--dump-eval` is Manaka's f32 reference (`examples/dump_eval` in `manaka-train`);
`--attic-dump` is this repository's `sym_dump` (for a `synn_*` shape) or
`nnue_dump` (for an `nnue_*` one) example, built for the shape the files are of:
`cargo build --release -p attic-eval --features <shape> --example sym_dump`.
`--mnw` is the trainer's file and `--nn` is what `manaka-train export-nnue`
wrote from it, quantised or with `--f32`.

Prints one line: the number of positions, then the largest and the mean
absolute difference in `value` and in `pre`, attic against the reference. Both
tools print the two columns, and for a `synn_*` shape each is one number, so the
two differences are the same; the script compares the columns as the tools give
them rather than assume it.

Standard library only, and no network file is read here: both programs read
their own. Nothing in this script is run by CI, since the weights it needs
(tens to hundreds of megabytes each) are not in the repository.
"""

import argparse
import subprocess
import sys


def rows(command):
    """The `(value, pre)` of each row of a tab-separated dump, in file order."""
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    if run.returncode != 0:
        sys.exit(f"{command[0]} failed ({run.returncode}):\n{run.stderr.strip()}")
    result = []
    for line in run.stdout.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        # Both tools lead with the index and the sfen, then `value`, then `pre`.
        result.append((int(fields[0]), fields[1], float(fields[2]), float(fields[3])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dump-eval", required=True)
    parser.add_argument("--attic-dump", required=True)
    parser.add_argument("--mnw", required=True)
    parser.add_argument("--nn", required=True)
    parser.add_argument("--sfens", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    limit = [] if args.limit is None else ["--limit", str(args.limit)]
    reference = rows(
        [args.dump_eval, "--weights", args.mnw, "--sfens", args.sfens, *limit]
    )
    attic = rows([args.attic_dump, "--net", args.nn, "--sfens", args.sfens, *limit])

    if len(reference) != len(attic):
        sys.exit(f"{len(reference)} reference rows against {len(attic)} from attic")
    value_diffs, pre_diffs = [], []
    for (i, sfen, rv, rp), (j, other, av, ap) in zip(reference, attic):
        if (i, sfen) != (j, other):
            sys.exit(f"row {i}: the two tools disagree on the position: {sfen!r} {other!r}")
        value_diffs.append(abs(av - rv))
        pre_diffs.append(abs(ap - rp))

    n = len(value_diffs)
    print(
        f"positions={n} "
        f"value: max={max(value_diffs):.3e} mean={sum(value_diffs) / n:.3e}  "
        f"pre: max={max(pre_diffs):.3e} mean={sum(pre_diffs) / n:.3e}"
    )


if __name__ == "__main__":
    main()
