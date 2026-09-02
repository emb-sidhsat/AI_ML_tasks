"""Command-line interface for standalone pipeline stages."""
import argparse

from src.pipeline.runner import run_all, run_phase

COMMANDS = {
    "ingest": "1",
    "preprocess": "2",
    "score": "3",
    "aggregate": "4",
    "validate": "5",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NHTSA Early Warning pipeline stages.")
    parser.add_argument("command", choices=(*COMMANDS, "run-all"), help="Pipeline stage to run.")
    parser.add_argument("--skip-semantic", action="store_true", help="Skip SBERT and zero-shot scoring.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run-all":
        run_all(skip_semantic=args.skip_semantic)
        return
    run_phase(COMMANDS[args.command], skip_semantic=args.skip_semantic if args.command == "score" else None)


if __name__ == "__main__":
    main()