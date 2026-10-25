import argparse
import json

from engine.game import Game


def load_teams(path="teams.json"):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"teams file not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"invalid JSON in {path}: {e}")


def _positive_ticks(value):
    try:
        ticks = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max-ticks must be a positive integer") from exc
    if ticks < 1:
        raise argparse.ArgumentTypeError("max-ticks must be a positive integer")
    return ticks


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a local full-map CTF match")
    parser.add_argument(
        "legacy_seed",
        nargs="?",
        type=int,
        help="match seed (legacy positional form; prefer --seed)",
    )
    parser.add_argument("--seed", type=int, help="deterministic map/match seed")
    parser.add_argument("--teams", default="teams.json", help="teams JSON file")
    parser.add_argument(
        "--max-ticks", type=_positive_ticks, default=6000,
        help="match duration in engine ticks (default: 6000)",
    )
    args = parser.parse_args(argv)
    if args.seed is not None and args.legacy_seed is not None:
        parser.error("provide either positional seed or --seed, not both")
    args.seed = args.seed if args.seed is not None else args.legacy_seed
    return args


def main(argv=None):
    args = _parse_args(argv)
    teams = load_teams(args.teams)
    game = Game(
        teams,
        spheres_per_team=3,
        seed=args.seed,
        max_ticks=args.max_ticks,
    )
    print(f"match started, telemetry file: {game.telemetry.file_path}")
    game.run()
    print(f"match over at tick {game.tick}, score: {game.score}")


if __name__ == "__main__":
    main()
