## Disparity Computation Procedure and Algorithm

### 1. **Procedure of Disparity Computing for Rectified Images**

Disparity computation involves calculating the pixel difference between two images captured from slightly different viewpoints. The goal is to estimate the depth of each pixel in the scene by comparing the horizontal displacement between corresponding pixels in the two images (stereo pair). Here’s the step-by-step procedure for computing the disparity from rectified images:

1. **Image Rectification**: The stereo images are rectified, ensuring that corresponding points between the two images are aligned horizontally. This simplifies the matching process to a one-dimensional search (along the horizontal axis).
   
2. **Block Matching**: For each pixel in the left image, a small block (or window) of pixels is selected around that pixel. This block is then compared with blocks of the same size in a range of positions along the corresponding row in the right image.

3. **Disparity Calculation**: The disparity is calculated as the horizontal shift (difference in x-coordinates) between the matching blocks in the left and right images. The pixel with the best match is identified based on a similarity measure (such as sum of absolute differences (SAD), normalized cross-correlation (NCC), or cost aggregation).

4. **Depth Estimation**: Since the disparity is inversely proportional to the depth, larger disparities correspond to objects closer to the camera, while smaller disparities correspond to objects farther away.

### 2. **Algorithm for Disparity Calculation (SAD Method)**

The following algorithm describes the disparity calculation using the Sum of Absolute Differences (SAD) method:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

BLOCK_SIZE = 7
SEARCH_BLOCK_SIZE = 56

def _read_image_pair(left_image_path="data/corridorl.jpg",
                     right_image_path="data/corridorr.jpg"):
    left_img = cv2.imread(left_image_path, 0)
    right_img = cv2.imread(right_image_path, 0)
    return np.asarray(left_img), np.asarray(right_img)

def calculate_sad(block_left, block_right):
    return np.sum(abs(block_left - block_right))

def compute_disparity_map_sad():
    left_array, right_array = _read_image_pair()
    left_array = left_array.astype(int)
    right_array = right_array.astype(int)
    h, w = left_array.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)

    for row in range(BLOCK_SIZE, h - BLOCK_SIZE):
        for col in range(BLOCK_SIZE, w - BLOCK_SIZE):
            block_left = left_array[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE]
            col_min = max(0, col - SEARCH_BLOCK_SIZE)
            col_max = col
            min_cost = float('inf')
            best_match = col
            for col_right in range(col_min, col_max):
                block_right = right_array[row:row + BLOCK_SIZE, col_right:col_right + BLOCK_SIZE]
                if block_right.shape != block_left.shape:
                    continue
                sad = calculate_sad(block_left, block_right)
                if sad < min_cost:
                    min_cost = sad
                    best_match = col_right
            disparity_map[row, col] = abs(best_match - col)

    disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_normalized = disparity_map_normalized.astype(np.uint8)

    plt.imshow(disparity_map_normalized, cmap='gray')
    plt.savefig('disparity_map_sad.png')
    plt.show()
```

### 3. **Application of the Algorithm and Observations on Disparity Maps**

I applied the developed algorithm using the SAD method to compute the disparity maps. Below are the corresponding results:

- **SAD Result**:
  The disparity map generated using the SAD method shows considerable differences in pixel intensity values, but lacks fine details and may have considerable noise. The result typically struggles in areas with repetitive textures or uniform regions, where finding good matches is challenging.

- **NCC and Cost Aggregation Results**:
  The disparity maps computed with these methods (see next sections) show improvement in quality, with fewer mismatches and better-defined object boundaries.

### 4. **Factors Affecting Disparity Map Computation and Algorithmic Improvements**

Several factors influence the quality of the computed disparity map:

1. **Window Size (Block Size)**: 
   A larger block size increases the reliability of the block matching but at the cost of reduced resolution. A smaller block size gives higher resolution but may lead to noisy results.

2. **Search Range (Disparity Range)**: 
   The search range defines how far to search for matching blocks between the left and right images. A wider range captures large disparities but increases computation time and may introduce mismatches.

3. **Similarity Measure (SAD, NCC, Cost Aggregation)**: 
   The measure used to compare pixel blocks significantly affects performance:
   - **SAD**: Simple and fast but sensitive to illumination changes and noise.
   - **NCC**: More robust to noise and illumination differences but computationally more expensive.
   - **Cost Aggregation**: Improves results by considering surrounding pixels during the matching process, reducing the noise and enhancing edge detection.

### 5. **Improvements: NCC and Cost Aggregation**

The following improvements were implemented and applied to the provided image pairs:

- **NCC (Normalized Cross-Correlation)**: 
   By using a normalized correlation coefficient, the matching process becomes less sensitive to brightness differences between the left and right images.

- **Cost Aggregation**: 
   This method aggregates matching costs over neighboring pixels to improve accuracy. It can reduce noise and improve consistency in textureless areas and around edges.

Here are the key improvements when using these methods:

1. **NCC Disparity Calculation**:
   ```python
   def calculate_ncc(block_left, block_right):
       mean_left = np.mean(block_left)
       mean_right = np.mean(block_right)
       numerator = np.sum((block_left - mean_left) * (block_right - mean_right))
       denominator = np.sqrt(np.sum((block_left - mean_left) ** 2) * np.sum((block_right - mean_right) ** 2))
       if denominator == 0:
           return -1
       return numerator / denominator
   ```

2. **Cost Aggregation Improvement**:
   ```python
   def compute_disparity_map_cost_aggregation():
       # Initialize the same as the SAD method, but include an aggregation window for improved results
       AGGREGATION_WINDOW_SIZE = 5
       ...
       # Similar to the NCC approach but aggregates over a window
       ncc_values = np.zeros(col - col_min + 1)
       for col_right in range(col_min, col_max):
           ...
           ncc = calculate_ncc(block_left, block_right)
           ncc_values[col_right - col_min] = ncc
   
       max_ncc = -1
       for i in range(len(ncc_values) - AGGREGATION_WINDOW_SIZE + 1):
           window = ncc_values[i:i + AGGREGATION_WINDOW_SIZE]
           aggregated_ncc = np.mean(window)
           if aggregated_ncc > max_ncc:
               max_ncc = aggregated_ncc
               best_match = col_min + i
   ```

These improvements lead to clearer, more defined disparity maps, especially in areas with repetitive textures or uniform regions.

### Conclusion

By applying different methods—SAD, NCC, and Cost Aggregation—on stereo image pairs, we observe that while SAD is the simplest and fastest approach, it struggles with noise and repetitive patterns. NCC provides robustness against lighting changes but requires more computational effort. Cost aggregation offers the most accurate results by considering surrounding pixel information, reducing noise and improving disparity estimation quality around object edges.

## Three Methods: SAD, NCC, and Cost Aggregation

These three methods—SAD, NCC, and Cost Aggregation—represent an evolution in stereo matching algorithms, progressing from simple to more complex approaches, from fast computation to higher accuracy. Below, I explain the evolution and improvements between these methods to help clarify their differences.

### 1. **SAD (Sum of Absolute Differences) — Basic Stereo Matching Method**

#### Basic Principle:
SAD is the simplest and most basic stereo matching algorithm. It works by comparing the absolute differences between pixel blocks in the left and right images to determine the best match. The steps are as follows:
- **Step 1**: For a given pixel in the left image, a fixed-size block (or window) around the pixel is selected.
- **Step 2**: A block of the same size is searched within the same row of the right image. The sum of absolute differences (SAD) is computed between the blocks.
- **Step 3**: The block with the smallest SAD is considered the best match.
- **Step 4**: Disparity is calculated as the horizontal shift (difference in x-coordinates) between the matched blocks in the left and right images.

#### Pros and Cons:
- **Pros**: Simple to implement and fast in computation.
- **Cons**: Highly sensitive to noise and brightness changes. It often results in mismatches in repetitive textures or textureless regions (e.g., smooth surfaces).

### 2. **NCC (Normalized Cross-Correlation) — Improvement Against Brightness Changes**

#### Improvement:
NCC improves upon SAD by being more robust to brightness differences between the left and right images. In NCC, pixel values within blocks are normalized before comparison, reducing the impact of illumination differences. The steps are as follows:
- **Step 1**: Similar to SAD, a block is selected in the left image, and a corresponding block is searched in the right image.
- **Step 2**: The average pixel values of both blocks are calculated, and these average values are subtracted from each pixel (normalization).
- **Step 3**: The normalized cross-correlation (NCC) is computed between the left and right blocks, and the block with the highest correlation is selected as the best match.
- **Step 4**: Disparity is calculated based on the horizontal shift between the matched blocks.

#### Pros and Cons:
- **Pros**: More robust than SAD and able to handle brightness differences, reducing mismatches.
- **Cons**: Computationally more expensive, especially on high-resolution images. It still struggles in textureless regions.

### 3. **Cost Aggregation — Enhanced Accuracy and Edge Processing**

#### Improvement:
Cost aggregation further improves upon NCC by aggregating matching costs over neighboring pixels, which reduces noise and improves the stability of matching. This method usually builds on top of NCC. The steps are:
- **Step 1**: Use NCC to compute the matching cost for individual pixels, as with the previous method.
- **Step 2**: Aggregate the costs for each pixel over a window of neighboring pixels (e.g., 3x3 or 5x5). The cost aggregation can be done by averaging or using more complex filtering methods.
- **Step 3**: After aggregating the costs, the best match is found using the aggregated cost values, and disparity is calculated.

#### Pros and Cons:
- **Pros**: By considering neighborhood information, cost aggregation significantly reduces isolated mismatches and performs better in textureless regions and around edges. The resulting disparity maps are smoother with less noise.
- **Cons**: Higher computational complexity compared to NCC alone, particularly when processing high-resolution images.

### Summary of the Evolution:

- **SAD Method**: This is the most basic matching method. It is suitable for quick implementation and computation but is highly sensitive to noise and brightness changes.
- **NCC Method**: Compared to SAD, NCC normalizes the pixel blocks, making it more robust to brightness variations. This improves matching accuracy and stability.
- **Cost Aggregation**: Cost aggregation builds upon NCC by considering surrounding pixels' information, reducing noise, and improving the accuracy of disparity maps, especially in textureless regions and edges.

### Conclusion:

1. **SAD** is a good choice for scenarios requiring fast computation but suffers from low accuracy and sensitivity to noise and brightness differences.
2. **NCC** is more robust to illumination changes, offering better matching accuracy and stability, making it suitable for complex image pairs.
3. **Cost Aggregation** improves NCC by aggregating costs over neighboring pixels, reducing noise and mismatch errors. It produces smoother disparity maps and is highly effective in practical applications.

This progression shows the trade-offs between computation speed, accuracy, and robustness to noise and brightness variations. The more complex methods, such as Cost Aggregation, are well-suited for tasks requiring high accuracy and robustness, even though they come with greater computational cost.