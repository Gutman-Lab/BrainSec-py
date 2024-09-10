"""
gray-white-matter-segmentation.py

This script is used to segment a single WSI file into gray and white matter regions. The
workflow used was adapted from "https://github.com/ucdrubinet/BrainSec". 

Usage:
    python gray-white-matter-segmentation.py --input-file <path-to-wsi-file> --save-dir <path-to-save-dir>
    
Arguments:
    --input-file: str: Path to the WSI file.
    --save-dir: str: Directory to save the output.
    --temp-dir: str: Temporary directory to save tiles.
    --tile-size: int: Size of the tiles to save.
    --stride: int: Stride of the sliding window.
    --num-workers: int: Number of workers to use for inference.
    --batch-size: int: Batch size for inference.
    --model-seg: str: Saved model for segmentation.

Dependencies:
    - numpy
    - pillow
    - scikit-image
    - torch
    - torchvision
    - tqdm
    
Author:
    Juan Carlos Vizcarra
    jvizcar@emory.edu
"""

from argparse import ArgumentParser
from time import perf_counter
from pathlib import Path
import numpy as np
from shutil import rmtree

from utils import tile_wsi, inference, cleanup_inference, save_label_mask_as_rgb


def arg_parse():
    # Parse command line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--input-file", type=str, required=True, help="Path to WSI file"
    )
    parser.add_argument(
        "--save-dir", type=str, required=True, help="Directory to save the output."
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=".temp",
        help="Temporary directory to save tiles",
    )
    parser.add_argument(
        "--tile-size", type=int, default=1536, help="Size of the tiles to save."
    )
    parser.add_argument(
        "--stride", type=int, default=16, help="Stride of the sliding window."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers to use for inference.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for inference."
    )
    parser.add_argument(
        "--model-seg",
        type=str,
        default="models/ResNet18_19.pkl",
        help="Saved model for segmentation.",
    )
    return parser.parse_args()


def main(args):
    # Start tracking the overall time.
    overall_start_time = perf_counter()
    
    # Tile the WSI file, saving the tiles to a directory
    temp_dir = Path(args.temp_dir)
    tile_dir = temp_dir / "tiles"

    start_time = perf_counter()  # log the start time of tiling
    tile_wsi(args.input_file, str(tile_dir), tile_size=args.tile_size)
    tiling_time = perf_counter() - start_time

    print(f"WSI file tiled in {tiling_time:.2f} seconds.")

    # Inference on the tiled WSI.
    start_time = perf_counter()  # log the start time of inference

    output_label_mask = inference(
        str(tile_dir / "0"),
        args.model_seg,
        img_size=args.tile_size,
        stride=args.stride,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    inference_time = perf_counter() - start_time

    print(f"Inference completed in {inference_time:.2f} seconds.")
    
    # Track the time for the cleanup process.
    start_time = perf_counter()
    
    # Create directory to save the output.
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the inference mask as a numpy.
    raw_label_mask_fp = save_dir / "raw_label_mask.npy"
    np.save(str(raw_label_mask_fp), output_label_mask)
    
    raw_rgb_mask_fp = save_dir / "raw_rgb_mask.png"
    save_label_mask_as_rgb(output_label_mask, str(raw_rgb_mask_fp))
    
    # Cleanup the inference mask.
    clean_output_label_mask = cleanup_inference(output_label_mask)
    
    # Save them as numpy and RGB.
    clean_label_mask_fp = save_dir / "clean_label_mask.npy"
    np.save(str(clean_label_mask_fp), clean_output_label_mask)
    
    clean_rgb_mask_fp = save_dir / "clean_rgb_mask.png"
    save_label_mask_as_rgb(clean_output_label_mask, str(clean_rgb_mask_fp))
    
    cleanup_time = perf_counter() - start_time
    
    print(f"Cleanup completed in {cleanup_time:.2f} seconds.")
    
    # Remove the temporary directory.
    rmtree(str(temp_dir))
    
    time_elapsed = perf_counter() - overall_start_time
    print(f"Total time elapsed: {time_elapsed:.2f} seconds.")


if __name__ == "__main__":
    main(arg_parse())
