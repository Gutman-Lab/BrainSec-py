"""
gray-white-matter-segmentation.py

This script is used to segment a single WSI file into gray and white matter regions. The
workflow used was adapted from "https://github.com/ucdrubinet/BrainSec". 

Usage:
    python gray-white-matter-segmentation.py --input-file <path-to-wsi-file>
    
Arguments:

Dependencies:
    -
    
Author:
    Juan Carlos Vizcarra
    jvizcar@emory.edu
"""
from argparse import ArgumentParser
from time import perf_counter
from pathlib import Path

from utils import tile_wsi, inference


def arg_parse():
    # Parse command line arguments.
    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, 
                        help="Path to WSI file")
    parser.add_argument("--temp-dir", type=str, default=".temp", 
                        help="Temporary directory to save tiles")
    parser.add_argument("--tile-size", type=int, default=1536, 
                        help="Size of the tiles to save.")
    parser.add_argument("--stride", type=int, default=16,
                        help="Stride of the sliding window.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers to use for inference.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference.")
    parser.add_argument("--model-seg", type=str, default='models/ResNet18_19.pkl', 
                        help="Saved model for segmentation.")
    # parser.add_argument("--normalize", type=bool, action="store_true",
    #                     help="Color normalize the tiles before saving")
    return parser.parse_args()


def main(args):
    # Tile the WSI file, saving the tiles to a directory
    save_dir = Path(args.temp_dir)
    tile_dir = save_dir / "tiles"
    
    # start_time = perf_counter()  # log the start time of tiling
    # tile_wsi(args.input_file, str(tile_dir), tile_size=args.tile_size)
    # tiling_time = perf_counter() - start_time
    
    # print(f"WSI file tiled in {tiling_time:.2f} seconds.")
    
    # Inference on the tiled WSI.
    start_time = perf_counter()  # log the start time of inference
    inference(
        str(tile_dir / "0"),
        args.model_seg,
        img_size=args.tile_size,
        stride=args.stride,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    

if __name__ == "__main__":
    main(arg_parse())