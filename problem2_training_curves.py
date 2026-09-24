#!/usr/bin/env python3
"""Plot recorded training history without training or loading a checkpoint."""
import argparse
import json
from pathlib import Path

from problem2.training_history import draw_training_curves, load_history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        payload = load_history(args.input_root)
        result = draw_training_curves(payload, args.output_root or args.input_root / "training_curves", dpi=args.dpi)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
