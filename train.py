"""Train a new NFS-e detector with the current Ultralytics architecture."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu or a CUDA device index")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset configuration does not exist: {args.data}")
    if args.epochs < 1 or args.imgsz < 32 or args.imgsz % 32:
        parser.error("Use positive epochs and an image size divisible by 32.")
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        seed=42,
        project="runs/train",
    )
    model.val(data=str(args.data.resolve()), device=args.device)


if __name__ == "__main__":
    main()
