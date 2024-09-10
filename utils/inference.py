import numpy as np
from torchvision import transforms
import os
from tqdm import tqdm
from utils.dataset import HeatmapDataset

import torch
from torch.autograd import Variable


def inference(
    img_dir, model, img_size: int = 1536, stride: int = 16, num_workers: int = 4, 
    batch_size: int = 8, color_norm_fp: str = "utils/normalization.npy"
) -> None:
    """Run inference on a set of tile images from a single WSI image. The input tile
    directory is created from the tile_wsi function (see tile_wsi.py).

    Args:
        img_size (int, optional): _description_. Defaults to 1536.
    """
    # Load color normalization parameters.
    norm = np.load(color_norm_fp, allow_pickle=True).item()
    normalize = transforms.Normalize(norm['mean'], norm['std'])  # as a transform
    to_tensor = transforms.ToTensor()
    
    # Load the model, only GPU use is supported.
    model = torch.load(model, map_location=lambda storage, loc: storage).cuda()         
    model.train(False)  # not training
        
    # Get a list of all the tile images.
    imgs = []
    
    for target in sorted(os.listdir(img_dir)):
        d = os.path.join(img_dir, target)
        
        if not os.path.isdir(d):
            # Skip non-directory files.
            continue
        
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if fname.endswith('.jpg'):
                    path = os.path.join(root, fname)
                    imgs.append(path)
           
    # Get the indices of rows and columns in the images, saved by tiles.         
    rows = [int(image.split('/')[-2]) for image in imgs]
    row_nums = max(rows) + 1
    cols = [int(image.split('/')[-1].split('.')[0]) for image in imgs]
    col_nums = max(cols) +1   
    
    # Create the output array according to the image size and stride used.
    heatmap_res = img_size // stride  # resolution of output heatmap
    seg_output = np.zeros((heatmap_res*row_nums, heatmap_res*col_nums), dtype=np.uint8)
        
    # Get a list of row / column indices.
    indices = [(row, col) for row in range(row_nums) for col in range(col_nums)]
    
    for idx in tqdm(indices, desc="Inferencing on each tile:"):
        row, col = idx

        image_datasets = HeatmapDataset(
            img_dir, 
            row, 
            col, 
            normalize, 
            stride=stride,
            img_size=img_size
        )
        
        dataloader = torch.utils.data.DataLoader(
            image_datasets, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers
        )
        
        # For Stride 32 (BrainSeg):
        running_seg = torch.zeros((32), dtype=torch.uint8)
        output_class = np.zeros((heatmap_res, heatmap_res), dtype=np.uint8)
        
        # Loop through the mini-batches.
        for idx, inputs in enumerate(dataloader):
            # wrap them in Variable
            with torch.no_grad():
                inputs = Variable(inputs.cuda())
            
                # Predict on the data.
                predict = model(inputs)
                
                # Indices = 0:Background, 1:WM, 2:GM.
                _, indices = torch.max(predict.data, 1)
                indices = indices.type(torch.uint8)
                running_seg =  indices.data.cpu()

                # For Stride 32 (BrainSeg) :
                i = (idx // (heatmap_res//batch_size))
                j = (idx % (heatmap_res//batch_size))
                output_class[i,j*batch_size:(j+1)*batch_size] = running_seg
        
                # Final Outputs of Brain Segmentation.
                seg_output[
                    row*heatmap_res:(row+1)*heatmap_res, 
                    col*heatmap_res:(col+1)*heatmap_res
                ] = output_class
