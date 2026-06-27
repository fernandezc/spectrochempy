"""Minimal CLI for the Scientific Workflow Assistant.

Usage:
    scp-ai explore DATA_PATH [--output OUTPUT] [--n-components N]
                              [--baseline-method METHOD] [--file-format FMT]

Only the exploratory_pca template is currently supported.
"""

from __future__ import annotations

import argparse
import sys

from spectrochempy_ai.exploration import explore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scp-ai",
        description="Scientific Workflow Assistant — create reproducible exploration notebooks",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    explore = sub.add_parser(
        "explore",
        help="Create an exploratory PCA notebook from a spectral dataset",
    )
    explore.add_argument(
        "input_path",
        help="Path to the spectral dataset file",
    )
    explore.add_argument(
        "--output", "-o",
        default=None,
        help="Output notebook path (default: <stem>-exploratory-pca.ipynb)",
    )
    explore.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Number of principal components (default: 5)",
    )
    explore.add_argument(
        "--baseline-method",
        default=None,
        help="Baseline correction method (default: asls)",
    )
    explore.add_argument(
        "--file-format",
        default=None,
        help="Input file format (default: scp)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "explore":
        try:
            result = explore(
                input_path=args.input_path,
                output_path=args.output,
                n_components=args.n_components,
                baseline_method=args.baseline_method,
                file_format=args.file_format,
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Notebook written to: {result}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
