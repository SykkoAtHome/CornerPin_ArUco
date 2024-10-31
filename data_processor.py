import numpy as np
import pandas as pd
from typing import Optional, Tuple


class DataProcessor:
    def __init__(self, data):
        """
        Initialize DataProcessor with Data object.

        Args:
            data: Data object containing markers information
        """
        self.data = data

    def calculate_relative_center(self, frame_index: int) -> Optional[Tuple[float, float]]:
        """
        Calculate the center of the rectangle formed by markers using the intersection
        of lines between opposite markers (ID0-ID2 and ID1-ID3).

        Args:
            frame_index: Index of the frame to process

        Returns:
            tuple: (center_x, center_y) if calculation successful
            None: if not enough markers or data incomplete
        """
        # Get data for specific frame
        frame_data = self.data.df[self.data.df['frame_index'] == frame_index]

        if frame_data.empty:
            print(f"No data found for frame {frame_index}")
            return None

        row = frame_data.iloc[0]

        # Get centers of markers
        marker_positions = {}
        for marker_id in range(4):
            x = row[f'id{marker_id}_center_x']
            y = row[f'id{marker_id}_center_y']

            if np.isnan(x) or np.isnan(y):
                print(f"Marker ID{marker_id} position not available")
                return None

            marker_positions[marker_id] = np.array([x, y])
            print(f"Marker ID{marker_id} position: ({x:.2f}, {y:.2f})")

        # Line 1: ID0 to ID2
        x1, y1 = marker_positions[0]  # ID0
        x2, y2 = marker_positions[2]  # ID2

        # Line 2: ID1 to ID3
        x3, y3 = marker_positions[1]  # ID1
        x4, y4 = marker_positions[3]  # ID3

        print(f"\nCalculating intersection between lines:")
        print(f"Line 1: ID0({x1:.2f}, {y1:.2f}) to ID2({x2:.2f}, {y2:.2f})")
        print(f"Line 2: ID1({x3:.2f}, {y3:.2f}) to ID3({x4:.2f}, {y4:.2f})")

        # Calculate intersection point using cross multiplication method
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if denominator == 0:
            print("Warning: Lines are parallel")
            return None

        numerator_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4))
        numerator_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4))

        intersection_x = numerator_x / denominator
        intersection_y = numerator_y / denominator

        print(f"Calculated intersection point: ({intersection_x:.2f}, {intersection_y:.2f})")

        # Basic sanity check - center should be within the bounding box of the markers
        min_x = min(x1, x2, x3, x4)
        max_x = max(x1, x2, x3, x4)
        min_y = min(y1, y2, y3, y4)
        max_y = max(y1, y2, y3, y4)

        if not (min_x <= intersection_x <= max_x and min_y <= intersection_y <= max_y):
            print("Warning: Calculated center point is outside markers bounding box!")

        return intersection_x, intersection_y
