import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import _read_image_pair

BLOCK_SIZE = 7
SEARCH_BLOCK_SIZE = 56

# Based on SAD
def calculate_sad(block_left, block_right):
    """
    Calculates the sum of the absolute differences by subtracting corresponding pixels in the two 
    blocks, taking absolute values and summing them
    """
    return np.sum(abs(block_left - block_right))

def compute_disparity_map_sad(args):
    """
    The function reads the left image and the right image paths and calculates diparity maps.
    It essentially picks up a block from left image and iterates through right image to find the best
    match by calculating the sum of absolute difference between pixels in left block and right block
    """
    left_array, right_array = _read_image_pair(args.left_image_path, args.right_image_path)
    left_array = left_array.astype(int)
    right_array = right_array.astype(int)
    h, w = left_array.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)

    for row in tqdm(range(BLOCK_SIZE, h - BLOCK_SIZE), desc='Process'):
        for col in range(BLOCK_SIZE, w - BLOCK_SIZE):
            block_left = left_array[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE] #extract block from left image

            #horizontal search range in right image
            col_min = max(0, col - SEARCH_BLOCK_SIZE)
            col_max = col
            min_cost = float('inf')
            best_match = col
            for col_right in range(col_min, col_max):
                block_right = right_array[row:row + BLOCK_SIZE, col_right:col_right + BLOCK_SIZE] #extract block from right image
                if block_right.shape != block_left.shape: # check block shape
                    continue
                sad = calculate_sad(block_left, block_right) #calculates SAD
                #update minimum cost and best match
                if sad < min_cost: 
                    min_cost = sad
                    best_match = col_right

            #calculate and store disparity
            disparity_map[row, col] = abs(best_match - col)

    #Normalize and store disparity match
    disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX) 
    disparity_map_normalized = disparity_map_normalized.astype(np.uint8)

    filename = args.left_image_path[:-5].split('/')[-1]
    plt.imshow(disparity_map_normalized, cmap='hot')
    plt.savefig(f'./output/{filename}_disparity_map_sad.png')
    plt.show()
