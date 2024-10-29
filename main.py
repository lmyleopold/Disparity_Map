import argparse
from sad_disparity import compute_disparity_map_sad
from ncc_disparity import compute_disparity_map_ncc
from cost_aggregation_disparity import compute_disparity_map_cost_aggregation
from weighted_median_filtering import compute_disparity_map_cost_aggregation_weighted_filtering
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Disparity Map Calculation with Different Methods')
    parser.add_argument('--method', type=str, choices=['SAD', 'NCC', 'CostAggregation', 'CostAggregation_WeightedFiltering', 'all'], default='SAD',
                        help='Choose the disparity map method: SAD, NCC, CostAggregation, CostAggregation_WeightedFiltering or for all')

    parser.add_argument('--left_image_path', type=str, default='data/corridorl.jpg',
                        help='The path to the left image')
    parser.add_argument('--right_image_path', type=str, default='data/corridorr.jpg',
                        help='The path to the right image')
    parser.add_argument('--all_images_flag', type=bool, default='False',
                        help='The path to the right image')
    parser.add_argument('--all_images', type=str, default='./data/',
                        help='The path to the right image')

    args = parser.parse_args()

    # Call different functions based on the selected algorithm

    if args.all_images_flag == True:
        processed_images = []
        for img in os.listdir(args.all_images):
            image_name = img[:-5]
            if image_name in processed_images:
                continue
            processed_images.append(image_name)
            args.left_image_path = args.all_images + image_name + "l.jpg"
            args.right_image_path = args.all_images + image_name + "r.jpg"

            if args.method == 'SAD':
                compute_disparity_map_sad(args)
            elif args.method == 'NCC':
                compute_disparity_map_ncc(args)
            elif args.method == 'CostAggregation':
                compute_disparity_map_cost_aggregation(args)
            elif args.method == 'CostAggregation_WeightedFiltering':
                compute_disparity_map_cost_aggregation_weighted_filtering(args)
            elif args.method == 'all':
                compute_disparity_map_sad(args)
                compute_disparity_map_ncc(args)
                compute_disparity_map_cost_aggregation(args)
                compute_disparity_map_cost_aggregation_weighted_filtering(args)
        
    else:   
        if args.method == 'SAD':
            compute_disparity_map_sad(args)
        elif args.method == 'NCC':
            compute_disparity_map_ncc(args)
        elif args.method == 'CostAggregation':
            compute_disparity_map_cost_aggregation(args)
        elif args.method == 'CostAggregation_WeightedFiltering':
            compute_disparity_map_cost_aggregation_weighted_filtering(args)
        elif args.method == 'all':
            compute_disparity_map_sad(args)
            compute_disparity_map_ncc(args)
            compute_disparity_map_cost_aggregation(args)
            compute_disparity_map_cost_aggregation_weighted_filtering(args)
