from pathlib import Path
import sys

from bhume import load, write_predictions
from bhume.baseline import global_median_shift


def main(village_dir):
    village = load(village_dir)

    preds = global_median_shift(village)

    out = write_predictions(
        Path(village_dir) / "predictions.geojson",
        preds
    )

    print(f"Predictions written to {out}")


if __name__ == "__main__":
    main(sys.argv[1])