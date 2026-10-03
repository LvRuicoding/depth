"""Run ``train``, ``eval-val``, ``smoke`` or ``summarize``."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "eval-val", "smoke", "summarize"))
    if not argv or argv[0] in ("-h", "--help"):
        parser.parse_args(argv)
        return
    command = parser.parse_args(argv[:1]).command
    if command == "summarize":
        from .summary import main as summary_main
        return summary_main(argv[1:])
    from .train import main as train_main
    return train_main(argv[1:], mode=command)


if __name__ == "__main__":
    main()
