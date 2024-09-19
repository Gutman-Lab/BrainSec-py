import pyvips as Vips
from pathlib import Path

def tile_wsi(fp: str, save_dir: str, tile_size: int = 1536) -> None:
    """Tile a WSI file and save the tiles to a specified directory.

    Args:
        fp (str): Input WSI filepath.
        save_dir (str): Directory to save the tile images.
        tile_size (int, optional): Size of the tiles to save. Defaults to 1536.
        
    Returns:
        None, saves tile images to save_dir.
        
    """
    # Use the pyvips library to read the WSI from file.
    vips_img = Vips.Image.new_from_file(fp, level=0)
    
    # Create save location.
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    Vips.Image.dzsave(
        vips_img, 
        save_dir,
        layout='google',
        suffix='.jpg[Q=90]',
        tile_size=tile_size,
        depth='one',
        properties=True
    )
    
    del vips_img  # clear up memory?