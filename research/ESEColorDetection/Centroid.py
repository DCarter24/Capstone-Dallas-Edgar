import cv2
import numpy as np
import datetime
import os

# Constants
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
path = "/home/pi/repo/Capstone-Dallas-Edgar/research/ESEColorDetection/PatchData"
crop_height = int(SCREEN_HEIGHT * 0.10)  # Adjusted to 48 pixels

camera = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

def getTime():
    return datetime.datetime.now().strftime("S_%S_M_%M")

for _ in range(1):  # Simplified loop
    camera.read()  # Discard the first frame
    successfulRead, raw_image = camera.read()
    if not successfulRead:
        print("Image not taken successful.")
        break

    cv2.imwrite(os.path.join(path, f"raw_image_{getTime()}.jpg"), raw_image)

    raw_image = cv2.flip(raw_image, -1)

    img_hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
    img_crop_hsv = img_hsv[crop_height:, :]

    lower_hsv = np.array([0, 0, 120])  # Adjusted for dark lighting
    upper_hsv = np.array([180, 50, 255])
    mask = cv2.inRange(img_crop_hsv, lower_hsv, upper_hsv)
    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    mask_edges = cv2.Canny(mask_blurred, 50, 150)

    crop_width = 20
    mask_edges = mask_edges[:, crop_width:]
    adjusted_screen_width = SCREEN_WIDTH - crop_width

    lines = cv2.HoughLinesP(mask_edges, 1, np.pi/180, 10, 12, 3)

    col_sum = np.sum(mask_edges > 0, axis=0)
    threshold = 50
    lane_columns = np.where(col_sum > threshold)[0]
    list_patch = []

    if lane_columns.size > 0:
        start = lane_columns[0]
        for c in lane_columns[1:]:
            if c != start + 1:
                list_patch.append((start, c - 1))
                start = c
        list_patch.append((start, lane_columns[-1]))

        num_patches_vertical = 4
        patch_height = (SCREEN_HEIGHT - crop_height) // num_patches_vertical
        patch_width = 20

        img_bottom_half_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        mask_edges_color = cv2.cvtColor(mask_edges, cv2.COLOR_GRAY2BGR)

        for (start, end) in list_patch:
            x0 = max(start - patch_width // 2, 0)
            x1 = min(end + patch_width // 2, adjusted_screen_width)
            for k in range(num_patches_vertical):
                y0 = k * patch_height
                y1 = (k + 1) * patch_height - 1
                roi = mask_edges[y0:y1, x0:x1]
                points = np.argwhere(roi > 0)
                if points.size > 0:
                    centroid = np.mean(points, axis=0) + [y0, x0]
                    centroid = centroid.astype(int)
                    print(f"Centroid found at {centroid} in patch starting at {x0}, {y0}")
                    cv2.circle(mask_edges_color, (centroid[1], centroid[0]), 3, (0, 165, 255), -1)

        cv2.imwrite(os.path.join(path, f"centroids_visualized_{getTime()}.jpg"), mask_edges_color)
        print("Centroids computed and visualized on mask edges.")
