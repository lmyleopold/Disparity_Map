# Disparity Map Computation and Improvement Techniques

## Project Overview

This project focuses on implementing and analyzing different methods for computing disparity maps from stereo image pairs. The project includes a basic Sum of Absolute Differences (SAD) algorithm, followed by more advanced techniques such as Normalized Cross-Correlation (NCC) and Cost Aggregation. The aim is to understand how different algorithms impact the quality of the disparity maps and explore ways to enhance depth estimation accuracy.

## Project Structure

- **data/**: Contains the stereo image pairs used for testing.
  - Original left and right stereo images.
  - Disparity maps generated using SAD, NCC, and Cost Aggregation.
- **report/**: Contains the final project report and documentation.
  - `project_report.pdf`: A report detailing the algorithms, results, and analysis.

## Getting Started

### Prerequisites

To run the code in this project, the following are required:

- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- tqdm (for progress bars)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lmyleopold/Disparity_Map.git
   ```
   
2. **Navigate to the project directory**:
   ```bash
   cd Disparity_Map
   ```

### Usage

The project includes a command-line tool for computing disparity maps using different methods. You can choose between SAD, NCC, or Cost Aggregation by specifying the method when running the script.

#### Install requirements
#### Command Structure

```bash
python main.py --left_image_path <left_image_path> --right_image_path <right_image_path> --method <SAD|NCC|CostAggregation>
```

#### Arguments

- `--image_path`: **(Required)** Paths to the left and right input images.
- `--method`: **(Required)** Disparity map computation method. Options are:
  - `SAD`: **(Default)** Sum of Absolute Differences.
  - `NCC`: Normalized Cross-Correlation.
  - `CostAggregation`: Disparity map with cost aggregation.

#### Example Commands

1. **SAD-based Disparity Map**:
   ```bash
   python main.py --left_image_path 'data/triclopsi2l.jpg' --right_image_path 'data/triclopsi2r.jpg' --method SAD
   ```

2. **NCC-based Disparity Map**:
   ```bash
   python main.py --left_image_path 'data/triclopsi2l.jpg' --right_image_path 'data/triclopsi2r.jpg' --method NCC
   ```

3. **Cost Aggregation Disparity Map**:
   ```bash
   python main.py --left_image_path 'data/triclopsi2l.jpg' --right_image_path 'data/triclopsi2r.jpg' --method CostAggregation
   ```

#### Output

- The computed disparity map will be saved in the specified output path or, by default, in the `data` folder with a suffix (`_sad`, `_ncc`, `_costagg`) corresponding to the method used.
- A window will plot and display the disparity map.

## Project Workflow

1. **Algorithm Implementation**:
   - **SAD**: The basic method using Sum of Absolute Differences to compute disparity maps.
   - **NCC**: An advanced method that calculates normalized cross-correlation to handle lighting variations and improve accuracy.
   - **Cost Aggregation**: Further enhancement using aggregated cost from neighboring pixels to provide smoother and more accurate disparity maps.
   
2. **Image Processing**:
   - Apply each algorithm to stereo image pairs.
   - Generate and visualize disparity maps for comparison.
   
3. **Result Analysis**:
   - Assess and compare the performance of each method.
   - Identify the advantages of advanced methods (NCC and Cost Aggregation) over the basic SAD method.

4. **Report Writing**:
   - Summarize the findings and performance of each method in a detailed report, including implementation insights and evaluation results.

## Results and Analysis

The disparity maps generated with each method demonstrate the progressive improvements from SAD to NCC to Cost Aggregation. 

- **SAD**: Works well for simple images but struggles with noise and textureless areas.
- **NCC**: Provides more robust results in scenes with varying lighting and textures.
- **Cost Aggregation**: Offers the best performance with smoother and more accurate disparity maps, especially around object boundaries.

For more detailed discussions of the results, refer to the [project report](report/project_report.pdf).

## Improvements and Future Work

- Further improvements can be made by incorporating techniques like Semi-Global Matching (SGM) and using machine learning methods to learn disparity directly from data.
- Future work could also involve real-time stereo vision applications using optimized versions of these algorithms.

This project demonstrates how various methods contribute to improved depth estimation in stereo vision, and the insights gained can be applied to more complex scenarios in computer vision. 