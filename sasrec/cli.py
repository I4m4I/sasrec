"""Command-line entry points. Required external files are checked first."""
import argparse
from pathlib import Path

from .config import load_config, require_files


def main():
    parser = argparse.ArgumentParser(description="SASRec source supplement; external experiment data required")
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--domain", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/sasrec.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config, args.domain)
        data_dir = args.data_root / args.domain
        output_dir = args.output_root / args.domain
        require_files(data_dir)
        if args.command == "prepare":
            from .background import prepare
            prepare(cfg, data_dir, output_dir)
            print("Background preparation complete.")
        else:
            from .trainer import evaluate_checkpoint, select_device, train
            device = select_device(args.device)
            if args.command == "train":
                train(cfg, data_dir, output_dir, device)
                print("Training and final evaluation complete; artifacts saved locally.")
            else:
                require_files(output_dir / "fit", ("best.pt",))
                evaluation = evaluate_checkpoint(output_dir / "fit" / "best.pt", data_dir, device)
                print("Test accuracy: {:.6f}".format(evaluation.accuracy))
    except (OSError, ValueError, KeyError, FloatingPointError) as exc:
        parser.error(str(exc))
