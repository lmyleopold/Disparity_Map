import argparse
from sad_disparity import compute_disparity_map_sad
from ncc_disparity import compute_disparity_map_ncc
from cost_aggregation_disparity import compute_disparity_map_cost_aggregation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Disparity Map Calculation with Different Methods')
    parser.add_argument('--method', type=str, choices=['SAD', 'NCC', 'CostAggregation'], default='SAD',
                        help='Choose the disparity map method: SAD, NCC, or CostAggregation')
    parser.add_argument('--left_image_path', type=str, default='data/corridorl.jpg',
                        help='The path to the left image')
    parser.add_argument('--right_image_path', type=str, default='data/corridorr.jpg',
                        help='The path to the right image')

    args = parser.parse_args()

    # Call different functions based on the selected algorithm
    if args.method == 'SAD':
        compute_disparity_map_sad(args)
    elif args.method == 'NCC':
        compute_disparity_map_ncc(args)
    elif args.method == 'CostAggregation':
        compute_disparity_map_cost_aggregation(args)
