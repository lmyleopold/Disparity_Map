import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import _read_image_pair

BLOCK_SIZE = 7
SEARCH_BLOCK_SIZE = 56
AGGREGATION_WINDOW_SIZE = 5

# Also based on NCC
def calculate_ncc(block_left, block_right):
    """
    Implements the formula for Normalized Cross Correlation to measure the similarity between two blocks of image
    """
    mean_left = np.mean(block_left)
    mean_right = np.mean(block_right)
    numerator = np.sum((block_left - mean_left) * (block_right - mean_right))
    denominator = np.sqrt(np.sum((block_left - mean_left) ** 2) * np.sum((block_right - mean_right) ** 2))
    if denominator == 0:
        return -1
    return numerator / denominator

def apply_weighted_median_filter(disparity_map, kernel_size=5):
    """
    Applies a weighted median filter to the disparity map using a Gaussian kernel 
    to reduce noise and preserve edges in the disparity map.
    """
    h, w = disparity_map.shape
    filtered_map = np.zeros_like(disparity_map)
    
    # Define the Gaussian weights for the weighted median filtering
    kernel = cv2.getGaussianKernel(kernel_size, sigma=1.0)
    weights = kernel @ kernel.T  # Gaussian kernel for 2D weights
    
    # Apply the weighted median filter
    for row in range(kernel_size // 2, h - kernel_size // 2):
        for col in range(kernel_size // 2, w - kernel_size // 2):
            # Extract a patch around the pixel
            patch = disparity_map[row - kernel_size // 2: row + kernel_size // 2 + 1,
                                  col - kernel_size // 2: col + kernel_size // 2 + 1]
            
            # Apply weights to the patch and calculate the weighted median
            flat_patch = patch.flatten()
            flat_weights = weights.flatten()
            sorted_idx = np.argsort(flat_patch)
            sorted_patch = flat_patch[sorted_idx]
            sorted_weights = flat_weights[sorted_idx]
            cumulative_sum = np.cumsum(sorted_weights)
            median_idx = np.searchsorted(cumulative_sum, cumulative_sum[-1] / 2)
            filtered_map[row, col] = sorted_patch[median_idx]
    
    return filtered_map

def compute_disparity_map_cost_aggregation_weighted_filtering(args):
    """
    The function reads the left image and the right image paths and calculates diparity maps.
    It essentially picks up a block from left image and iterates through right image to find the best
    match by calculating NCC between left block and right block and then aggregating the NCC costs over
    a fixed window size around each pixel to smooth out noise. The maximum aggregated NCC score is chosen as the best match for the 
    left block, and the corresponding column (best_match) in the right image is recorded. Furthermore, the Weighted Median Filtering is applied 
    on the final disparity map. This technique smooths the disparity map output while preserving edges, 
    which is useful in reducing noise around object boundaries.
    """
    left_array, right_array = _read_image_pair(args.left_image_path, args.right_image_path)
    left_array = left_array.astype(int)
    right_array = right_array.astype(int)
    h, w = left_array.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)

    for row in tqdm(range(BLOCK_SIZE, h - BLOCK_SIZE), desc='Progress'):
        for col in range(BLOCK_SIZE, w - BLOCK_SIZE):
            block_left = left_array[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE]  #extract block from left image
            #horizontal search range in right image
            col_min = max(0, col - SEARCH_BLOCK_SIZE)
            col_max = col
            ncc_values = np.zeros(col - col_min + 1)
            for col_right in range(col_min, col_max):
                block_right = right_array[row:row + BLOCK_SIZE, col_right:col_right + BLOCK_SIZE] #extract block from right image
                if block_right.shape != block_left.shape: # check block shape
                    continue
                ncc = calculate_ncc(block_left, block_right) #calculates NCC
                ncc_values[col_right - col_min] = ncc

            max_ncc = -1
            best_match = col
            for i in range(len(ncc_values) - AGGREGATION_WINDOW_SIZE + 1):
                window = ncc_values[i:i + AGGREGATION_WINDOW_SIZE] # slides window across ncc values
                aggregated_ncc = np.mean(window) #mean of ncc values
                if aggregated_ncc > max_ncc:
                    max_ncc = aggregated_ncc
                    best_match = col_min + i
            #maximum aggregaed ncc value is chosen as best match
            disparity_map[row, col] = abs(best_match - col)
            
    
    
    #Normalize and store disparity match
    disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_normalized = disparity_map_normalized.astype(np.uint8)

    # median filtered disparity map
    median_filtered_disparity_map = apply_weighted_median_filter(disparity_map=disparity_map_normalized, kernel_size=7)

    filename = args.left_image_path[:-5].split('/')[-1]
    plt.imshow(median_filtered_disparity_map, cmap='grey')
    plt.savefig(f'./output/{filename}_disparity_map_costagg_weighted._filtering.png')
    plt.show()

