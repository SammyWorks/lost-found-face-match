import cv2
import numpy as np

def draw_bounding_box(image, box):
    img = np.array(image)
    x, y, w, h = box

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )
    return img
