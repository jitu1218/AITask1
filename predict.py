"""Predict the label of a centered, light-on-dark handwritten digit."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from model import DigitCNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--invert", action="store_true", help="For dark ink on a light background")
    args = parser.parse_args()
    with Image.open(args.image) as image:
        gray = ImageOps.grayscale(image)
        if args.invert:
            gray = ImageOps.invert(gray)
        square = ImageOps.pad(gray, (8, 8), method=Image.Resampling.LANCZOS, color=0)
        pixels = np.asarray(square, dtype=np.float32) / 255
    model = DigitCNN()
    path = Path(__file__).resolve().parent / "results" / "best_model.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(pixels)[None, None])
    print(f"Predicted digit: {logits.argmax(1).item()}")


if __name__ == "__main__":
    main()
