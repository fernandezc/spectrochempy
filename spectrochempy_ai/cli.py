"""
Minimal CLI for the Scientific Workflow Assistant.

Usage:
    scp-ai explore DATA_PATH [--output OUTPUT] [--n-components N]
                              [--baseline-method METHOD] [--file-format FMT]
                              [--reference REF_PATH]
    scp-ai suggest DATA_PATH [--reference REF_PATH] [--intent INTENT]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from spectrochempy_ai.exploration import explore
from spectrochempy_ai.rule_planner import suggest


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
        "--output",
        "-o",
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
    explore.add_argument(
        "--reference",
        "-r",
        default=None,
        help="Path to reference values file (for pls_calibration template)",
    )
    explore.add_argument(
        "--open",
        "-O",
        action="store_true",
        default=False,
        help="Open the notebook in Jupyter Lab after creation",
    )

    suggest = sub.add_parser(
        "suggest",
        help="Suggest workflow templates for a spectral dataset",
    )
    suggest.add_argument(
        "input_path",
        help="Path to the spectral dataset file",
    )
    suggest.add_argument(
        "--reference",
        "-r",
        default=None,
        help="Path to reference values file (for PLS calibration)",
    )
    suggest.add_argument(
        "--intent",
        default=None,
        choices=["explore", "calibrate", "resolve"],
        help="Scientific intent hint (explore, calibrate, resolve)",
    )
    return parser


def _open_notebook(path: Path) -> None:
    """Try to open a notebook in Jupyter Lab."""
    jupyter = shutil.which("jupyter")
    if jupyter is None:
        print("note: jupyter not found on PATH", file=sys.stderr)
        print(f"Open with: jupyter lab {path}")
        return
    try:
        subprocess.Popen(
            [jupyter, "lab", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        print("note: could not launch Jupyter Lab", file=sys.stderr)
        print(f"Open with: jupyter lab {path}")


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
                reference_path=args.reference,
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Notebook written to: {result}")
        if args.open:
            _open_notebook(result)
    elif args.command == "suggest":
        recommendations = suggest(
            args.input_path,
            reference_path=args.reference,
            intent=args.intent,
        )
        if not recommendations:
            print("No template suggestions available.", file=sys.stderr)
            sys.exit(1)
        for i, rec in enumerate(recommendations, 1):
            _print_recommendation(i, rec)
    else:
        parser.print_help()
        sys.exit(1)


def _print_recommendation(
    index: int,
    rec: TemplateRecommendation,  # noqa: F821 — forward reference resolved at runtime
) -> None:
    from spectrochempy_ai.rule_planner import TemplateRecommendation

    # Re-assert for type narrowing
    if not isinstance(rec, TemplateRecommendation):
        return

    tid = rec.template_id or "(no template)"
    print(f"\n[{index}] {tid}  (confidence: {rec.confidence:.2f})")
    print(f"  Dataset: {rec.dataset_summary}")
    print()
    print(rec.explain())


if __name__ == "__main__":
    main()
