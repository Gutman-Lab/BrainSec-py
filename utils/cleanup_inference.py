import numpy as np
from PIL import Image
from skimage import morphology


def _swap_GM_WM(arr):
    """Swap GM and WM in arr (swaps index 1 and index 2)"""
    arr_1 = (arr == 1)
    arr[arr == 2] = 1
    arr[arr_1] = 2
    del arr_1
    
    return arr
    

def cleanup_inference(label_mask: np.ndarray, down_factor: int = 4) -> np.ndarray:
    """Cleanup the gray / white matter inference mask.
    
    Args:
        label_mask (np.ndarray): The inference mask.
        down_factor (int, optional): The down factor of the mask. Defaults to 4.
        
    Returns:
        np.ndarray: The cleaned mask.
        
    """
    label_mask = Image.fromarray(label_mask)  # convert to PIL image
    
    width, height = label_mask.width, label_mask.height
    area_threshold_prop = 0.05
    area_threshold = int(area_threshold_prop * width * height // down_factor**2)
    
    # Downsample the image
    mask_arr = np.array(
        label_mask.resize((width // down_factor, height // down_factor), Image.NEAREST))
    
    del label_mask  # why delete this variable?
    
    # Apply area_opening to remove local maxima with area < 20000 px.
    mask_arr = morphology.area_opening(mask_arr, area_threshold=3200 // down_factor**2)
    
    # Swap index of GM and WM.
    mask_arr = _swap_GM_WM(mask_arr)
    
    # Apply area_opening to remove local maxima with area < 20000 px.
    mask_arr = morphology.area_opening(mask_arr, area_threshold=3200 // down_factor**2)
    
    # Swap index back.
    mask_arr = _swap_GM_WM(mask_arr)
    
    # Apply area_closing to remove local minima with area < 12500 px
    mask_arr = morphology.area_closing(mask_arr, area_threshold=2000 // down_factor**2)

    # Apply remove_small_objects to remove tissue residue with area < 0.05 * width * height
    tissue_arr = morphology.remove_small_objects(mask_arr > 0, min_size=area_threshold,
                                                 connectivity=2)
    mask_arr[np.invert(tissue_arr)] = 0
    del tissue_arr

    # Apply opening with disk-shaped kernel (r=8) to smooth boundary
    mask_arr = morphology.opening(mask_arr, footprint=morphology.disk(radius=32 // down_factor))

    # Upsample the output
    mask_arr = np.array(Image.fromarray(mask_arr).resize((width, height), Image.NEAREST))

    return mask_arr