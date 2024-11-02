import cv2
from image import Image
from detector import ArucoDetector
from data import Data
from draw import Draw, MarkerElements
import pandas as pd
import os

# Initialize components
name = "testy"
EXPECTED_MARKERS = 4
img_loader = Image('img/fix_alignment/')
detector = ArucoDetector(expected_markers=EXPECTED_MARKERS)
data = Data(expected_markers=EXPECTED_MARKERS)

# Initialize Draw with configuration
draw = Draw()
elements = MarkerElements(
    outline=True,
    corners=True,
    corner_numbers=True,
    center=True,
    id=False,
    orientation=True,
    angle=True
)
draw.set_marker_elements(elements)
draw.set_marker_visibility([0, 1, 2, 3])  # Show all markers

# Test on a single image
frame, index = img_loader.get_frame_by_index(5)
success = detector.detect(frame, data, index)

if success:
    # Visualization based on DataFrame data
    img_markers = draw.draw_markers(frame, data, index)

    # Display DataFrame
    print("\nCollected data:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(data.df)

    # Display image
    cv2.imshow(f'Frame {index} with markers', img_markers)
    print("\nPress any key to proceed to next visualization...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
