import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import _read_image_pair

BLOCK_SIZE = 7
SEARCH_BLOCK_SIZE = 56

# Based on NCC
def calculate_ncc(block_left, block_right):
    mean_left = np.mean(block_left)
    mean_right = np.mean(block_right)
    numerator = np.sum((block_left - mean_left) * (block_right - mean_right))
    denominator = np.sqrt(np.sum((block_left - mean_left) ** 2) * np.sum((block_right - mean_right) ** 2))
    if denominator == 0:
        return -1
    return numerator / denominator

def compute_disparity_map_ncc(args):
    left_array, right_array = _read_image_pair(args.left_image_path, args.right_image_path)
    left_array = left_array.astype(int)
    right_array = right_array.astype(int)
    h, w = left_array.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)

    for row in tqdm(range(BLOCK_SIZE, h - BLOCK_SIZE), desc='Progress'):
        for col in range(BLOCK_SIZE, w - BLOCK_SIZE):
            block_left = left_array[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE]
            col_min = max(0, col - SEARCH_BLOCK_SIZE)
            col_max = col
            max_ncc = -1
            best_match = col
            for col_right in range(col_min, col_max):
                block_right = right_array[row:row + BLOCK_SIZE, col_right:col_right + BLOCK_SIZE]
                if block_right.shape != block_left.shape:
                    continue
                ncc = calculate_ncc(block_left, block_right)
                if ncc > max_ncc:
                    max_ncc = ncc
                    best_match = col_right
            disparity_map[row, col] = abs(best_match - col)

    disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_normalized = disparity_map_normalized.astype(np.uint8)

    plt.imshow(disparity_map_normalized, cmap='gray')
    plt.savefig(f'{args.left_image_path[:-5]}_disparity_map_ncc.png')
    plt.show()
