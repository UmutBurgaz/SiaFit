"""Command-line interface for Si II and Ca II NIR fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import FitConfig


def _add_catalog_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("catalog", type=Path, help="Input CSV catalogue")
    parser.add_argument(
        "--features", choices=("si", "ca", "all"), default="si",
        help="Fit Si II, Ca II NIR, or Si II followed by Ca II NIR",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("siafit_results"), help="Output directory"
    )
    parser.add_argument(
        "--spectra-dir",
        type=Path,
        default=None,
        help="Fallback directory for spectrum filenames",
    )
    parser.add_argument("--start", type=int, default=0, help="First catalogue row")
    parser.add_argument("--end", type=int, default=None, help="Stop before this row")
    parser.add_argument(
        "--resume", action="store_true", help="Skip record IDs already in the output"
    )


def _add_preparation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--si-velocity", type=float, default=None,
        help="Si II 6355 velocity in 10³ km/s for Ca-only fitting; catalogue values take precedence",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "flux", "flux-error"),
        default="auto",
        help="Default input columns; a catalogue input_format column overrides this",
    )
    parser.add_argument(
        "--vexp",
        type=float,
        default=None,
        help="Smoothing sigma/lambda; default estimates it from S/N",
    )
    parser.add_argument("--no-clip", action="store_true", help="Disable automatic outlier clipping")


def _add_background_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=("auto", "review", "manual"),
        default="auto",
        help="Default mode; a catalogue mode column overrides this per spectrum",
    )
    parser.add_argument(
        "--vel-width",
        type=float,
        default=4.0,
        help="Continuum-window width in 10^3 km/s",
    )


def _add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--niter", type=int, default=100, help="Monte Carlo draws; 0 disables MC")
    parser.add_argument(
        "--mc-mode",
        choices=("resample", "shift", "shift_resample", "all"),
        default="resample",
    )
    parser.add_argument(
        "--continuum-shift",
        type=float,
        default=1.3,
        help="Anchor-shift sigma in 10^3 km/s for shift-based MC modes",
    )
    parser.add_argument("--seed", type=int, default=None, help="Reproducible base seed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="siafit",
        description="Fit Si II 6355/5972 and Ca II NIR features in supernova spectra",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.2.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Select backgrounds and fit spectra end to end")
    _add_catalog_arguments(run)
    _add_preparation_arguments(run)
    _add_background_arguments(run)
    _add_fit_arguments(run)

    background = subparsers.add_parser(
        "background", help="Select and save backgrounds without Monte Carlo fitting"
    )
    _add_catalog_arguments(background)
    _add_preparation_arguments(background)
    _add_background_arguments(background)

    fit = subparsers.add_parser("fit", help="Fit spectra from a saved backgrounds.csv catalogue")
    _add_catalog_arguments(fit)
    _add_preparation_arguments(fit)
    _add_fit_arguments(fit)
    fit.add_argument(
        "--vel-width",
        type=float,
        default=4.0,
        help="Fallback only; saved per-row widths take precedence",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> FitConfig:
    return FitConfig(
        mode=getattr(args, "mode", "auto"),
        features=args.features,
        si_velocity=args.si_velocity,
        input_format=args.input_format,
        vexp=args.vexp,
        clip_outliers=not args.no_clip,
        vel_width=args.vel_width,
        n_iterations=getattr(args, "niter", 0),
        mc_mode=getattr(args, "mc_mode", "resample"),
        continuum_shift=getattr(args, "continuum_shift", 1.3),
        seed=getattr(args, "seed", None),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)

    # Defer the heavier scientific imports until after argument parsing, so
    # `siafit --help` remains fast and useful even during environment setup.
    from .pipeline import fit_background_catalog, run_catalog

    common = dict(
        config=config,
        spectra_dir=args.spectra_dir,
        start=args.start,
        end=args.end,
        resume=args.resume,
    )
    if args.command == "fit":
        summary = fit_background_catalog(args.catalog, args.output_dir, **common)
    else:
        summary = run_catalog(
            args.catalog,
            args.output_dir,
            stage="all" if args.command == "run" else "background",
            **common,
        )

    print(
        f"Finished: {summary.succeeded} succeeded, {summary.failed} failed, "
        f"{summary.skipped} skipped. Output: {summary.output_dir}"
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
