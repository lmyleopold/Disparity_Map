import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import _read_image_pair

BLOCK_SIZE = 5
SEARCH_BLOCK_SIZE = 56

def compute_disparity_map_semi_global_matching(args):
    """
    The function reads the left image and the right image paths and calculates diparity maps.
    It essentially picks up a block from left image and iterates through right image to find the best
    match by calculating NCC between left block and right block
    """
    left_array, right_array = _read_image_pair(args.left_image_path, args.right_image_path)
    left_array = left_array.astype(np.uint8)
    right_array = right_array.astype(np.uint8)
    h, w = left_array.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)

    # stereo = cv2.StereoBM.create(numDisparities = 16, blockSize=BLOCK_SIZE)
    stereo = cv2.StereoSGBM_create( blockSize=BLOCK_SIZE)

    # Set other parameters
    # stereo.setPreFilterType(1)  # Median filtering
    # stereo.setPreFilterSize(5)   # Pre-filter size
    # stereo.setTextureThreshold(10)
    # stereo.setUniquenessRatio(15)
    disparity_map = stereo.compute(left_array, right_array)

    #Normalize and store disparity match
    disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_normalized = disparity_map_normalized.astype(np.uint8)

    filename = args.left_image_path[:-5].split('/')[-1]
    plt.imshow(disparity_map_normalized, cmap='grey')
    plt.savefig(f'./output/{filename}_disparity_map_sgm2.png')
    plt.show()