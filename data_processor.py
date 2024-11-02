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

    def calculate_aspect_ratio(self, frame_index: int) -> dict:
        """
        Calculate aspect ratio of the marker rectangle.
        Returns both outer and inner rectangle ratios.

        Args:
            frame_index: Frame number to analyze

        Returns:
            dict with calculated ratios and distances:
            {
                'outer_ratio': float,  # width/height for outer points
                'inner_ratio': float,  # width/height for inner points
                'average_ratio': float,  # average of both ratios
                'outer_width': float,  # width of outer rectangle
                'outer_height': float,  # height of outer rectangle
                'inner_width': float,  # width of inner rectangle
                'inner_height': float  # height of inner rectangle
            }
        """
        frame_data = self.data.df[self.data.df['frame_index'] == frame_index]
        if frame_data.empty:
            print(f"No data found for frame {frame_index}")
            return None

        row = frame_data.iloc[0]

        # Get outer rectangle points
        outer_points = {
            'top_left': (row['id0_corner0_x'], row['id0_corner0_y']),  # ID0 corner 0
            'top_right': (row['id1_corner1_x'], row['id1_corner1_y']),  # ID1 corner 1
            'bottom_right': (row['id2_corner2_x'], row['id2_corner2_y']),  # ID2 corner 2
            'bottom_left': (row['id3_corner3_x'], row['id3_corner3_y'])  # ID3 corner 3
        }

        # Get inner rectangle points
        inner_points = {
            'top_left': (row['id0_corner2_x'], row['id0_corner2_y']),  # ID0 corner 2
            'top_right': (row['id1_corner3_x'], row['id1_corner3_y']),  # ID1 corner 3
            'bottom_right': (row['id2_corner0_x'], row['id2_corner0_y']),  # ID2 corner 0
            'bottom_left': (row['id3_corner1_x'], row['id3_corner1_y'])  # ID3 corner 1
        }

        # Calculate distances for outer rectangle
        outer_width_top = self._calculate_distance(
            outer_points['top_left'], outer_points['top_right'])
        outer_width_bottom = self._calculate_distance(
            outer_points['bottom_left'], outer_points['bottom_right'])
        outer_height_left = self._calculate_distance(
            outer_points['top_left'], outer_points['bottom_left'])
        outer_height_right = self._calculate_distance(
            outer_points['top_right'], outer_points['bottom_right'])

        # Calculate distances for inner rectangle
        inner_width_top = self._calculate_distance(
            inner_points['top_left'], inner_points['top_right'])
        inner_width_bottom = self._calculate_distance(
            inner_points['bottom_left'], inner_points['bottom_right'])
        inner_height_left = self._calculate_distance(
            inner_points['top_left'], inner_points['bottom_left'])
        inner_height_right = self._calculate_distance(
            inner_points['top_right'], inner_points['bottom_right'])

        # Average the widths and heights
        outer_width = (outer_width_top + outer_width_bottom) / 2
        outer_height = (outer_height_left + outer_height_right) / 2
        inner_width = (inner_width_top + inner_width_bottom) / 2
        inner_height = (inner_height_left + inner_height_right) / 2

        # Calculate ratios
        outer_ratio = outer_width / outer_height
        inner_ratio = inner_width / inner_height
        average_ratio = (outer_ratio + inner_ratio) / 2

        print(f"\nAspect Ratio Analysis for frame {frame_index}:")
        print(f"Outer rectangle - Width: {outer_width:.2f}, Height: {outer_height:.2f}, Ratio: {outer_ratio:.6f}")
        print(f"Inner rectangle - Width: {inner_width:.2f}, Height: {inner_height:.2f}, Ratio: {inner_ratio:.6f}")
        print(f"Average aspect ratio: {average_ratio:.6f}")

        return {
            'outer_ratio': outer_ratio,
            'inner_ratio': inner_ratio,
            'average_ratio': average_ratio,
            'outer_width': outer_width,
            'outer_height': outer_height,
            'inner_width': inner_width,
            'inner_height': inner_height
        }

    def _calculate_distance(self, point1: tuple, point2: tuple) -> float:
        """
        Calculate Euclidean distance between two points
        """
        from math import sqrt
        x1, y1 = point1
        x2, y2 = point2
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
