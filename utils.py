import cv2
import numpy as np

def _read_image_pair(left_image_path="data/corridorl.jpg",
                     right_image_path="data/corridorr.jpg"):
    left_img = cv2.imread(left_image_path, 0)
    right_img = cv2.imread(right_image_path, 0)
    return np.asarray(left_img), np.asarray(right_img)