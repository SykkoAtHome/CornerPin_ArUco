import cv2
from image import Image
from detector import ArucoDetector
from data import Data
import pandas as pd
import os

# Inicjalizacja
name = "testy"
EXPECTED_MARKERS = 4
img_loader = Image('img/fix_alignment/')
detector = ArucoDetector(expected_markers=EXPECTED_MARKERS)
# data = Data(expected_markers=EXPECTED_MARKERS, file_path=f"{name}/{name}_detections.csv")
data = Data(expected_markers=EXPECTED_MARKERS)

# Test on a single image
frame, index = img_loader.get_frame_by_index(3)
success = detector.detect(frame, data, index)

if success:
    # Visualization based on DataFrame data
    img_markers = img_loader.draw_markers(frame, data, index)

    # Display DataFrame
    print("\nCollected data:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(data.df)

    # Display image
    cv2.imshow(f'Frame {index} with markers', img_markers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
