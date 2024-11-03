import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List


class DataProcessor:
    def __init__(self, data):
        """
        Initialize DataProcessor with Data object.

        Args:
            data: Data object containing markers information
        """
        self.data = data

    def _calculate_distance(self, point1: tuple, point2: tuple) -> float:
        """
        Calculate Euclidean distance between two points.

        Args:
            point1: First point as (x,y) tuple
            point2: Second point as (x,y) tuple

        Returns:
            float: Euclidean distance between points
        """
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _get_marker_centers(self, row) -> Dict[int, Tuple[float, float]]:
        """
        Get centers of all markers from a DataFrame row.

        Args:
            row: DataFrame row containing marker data

        Returns:
            dict: Dictionary with marker IDs as keys and (x,y) tuples as values
        """
        centers = {}
        for marker_id in range(4):
            x = row[f'id{marker_id}_center_x']
            y = row[f'id{marker_id}_center_y']
            centers[marker_id] = (x, y)
        return centers

    def _get_rectangle_points(self, row) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Get outer and inner rectangle points from a DataFrame row.

        Args:
            row: DataFrame row containing marker data

        Returns:
            dict: Dictionary containing outer and inner rectangle points
        """
        outer_points = {
            'top_left': (row['id0_corner0_x'], row['id0_corner0_y']),
            'top_right': (row['id1_corner1_x'], row['id1_corner1_y']),
            'bottom_right': (row['id2_corner2_x'], row['id2_corner2_y']),
            'bottom_left': (row['id3_corner3_x'], row['id3_corner3_y'])
        }

        inner_points = {
            'top_left': (row['id0_corner2_x'], row['id0_corner2_y']),
            'top_right': (row['id1_corner3_x'], row['id1_corner3_y']),
            'bottom_right': (row['id2_corner0_x'], row['id2_corner0_y']),
            'bottom_left': (row['id3_corner1_x'], row['id3_corner1_y'])
        }

        return {'outer': outer_points, 'inner': inner_points}

    def _calculate_rectangle_dimensions(self, points: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Calculate rectangle width and height from corner points.

        Args:
            points: Dictionary with corner points

        Returns:
            dict: Dictionary with width and height measurements
        """
        width_top = self._calculate_distance(points['top_left'], points['top_right'])
        width_bottom = self._calculate_distance(points['bottom_left'], points['bottom_right'])
        height_left = self._calculate_distance(points['top_left'], points['bottom_left'])
        height_right = self._calculate_distance(points['top_right'], points['bottom_right'])

        return {
            'width': (width_top + width_bottom) / 2,
            'height': (height_left + height_right) / 2
        }

    def calculate_relative_center(self, frame_index: int) -> Optional[Tuple[float, float]]:
        """
        Calculate the center of the rectangle formed by markers.
        """
        frame_data = self.data.df[self.data.df['frame_index'] == frame_index]
        if frame_data.empty:
            print(f"No data found for frame {frame_index}")
            return None

        centers = self._get_marker_centers(frame_data.iloc[0])
        if None in centers.values():
            print("Some marker positions not available")
            return None

        # Line 1: ID0 to ID2
        x1, y1 = centers[0]
        x2, y2 = centers[2]

        # Line 2: ID1 to ID3
        x3, y3 = centers[1]
        x4, y4 = centers[3]

        print(f"\nCalculating intersection between lines:")
        print(f"Line 1: ID0({x1:.2f}, {y1:.2f}) to ID2({x2:.2f}, {y2:.2f})")
        print(f"Line 2: ID1({x3:.2f}, {y3:.2f}) to ID3({x4:.2f}, {y4:.2f})")

        # Calculate intersection point
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            print("Warning: Lines are parallel")
            return None

        numerator_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4))
        numerator_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4))

        intersection_x = numerator_x / denominator
        intersection_y = numerator_y / denominator

        # Verify point is within bounds
        if not (min(x1, x2, x3, x4) <= intersection_x <= max(x1, x2, x3, x4) and
                min(y1, y2, y3, y4) <= intersection_y <= max(y1, y2, y3, y4)):
            print("Warning: Calculated center point is outside markers bounding box!")

        return intersection_x, intersection_y

    def calculate_aspect_ratio(self, frame_index: int) -> Optional[Dict[str, float]]:
        """
        Calculate aspect ratio of the marker rectangle.
        """
        frame_data = self.data.df[self.data.df['frame_index'] == frame_index]
        if frame_data.empty:
            print(f"No data found for frame {frame_index}")
            return None

        rectangles = self._get_rectangle_points(frame_data.iloc[0])

        # Calculate dimensions for both rectangles
        outer_dims = self._calculate_rectangle_dimensions(rectangles['outer'])
        inner_dims = self._calculate_rectangle_dimensions(rectangles['inner'])

        # Calculate ratios
        outer_ratio = outer_dims['width'] / outer_dims['height']
        inner_ratio = inner_dims['width'] / inner_dims['height']
        average_ratio = (outer_ratio + inner_ratio) / 2

        # print(f"\nAspect Ratio Analysis for frame {frame_index}:")
        # print(
        #     f"Outer rectangle - Width: {outer_dims['width']:.2f}, Height: {outer_dims['height']:.2f}, Ratio: {outer_ratio:.6f}")
        # print(
        #     f"Inner rectangle - Width: {inner_dims['width']:.2f}, Height: {inner_dims['height']:.2f}, Ratio: {inner_ratio:.6f}")
        # print(f"Average aspect ratio: {average_ratio:.6f}")

        return {
            'outer_ratio': outer_ratio,
            'inner_ratio': inner_ratio,
            'average_ratio': average_ratio,
            'outer_width': outer_dims['width'],
            'outer_height': outer_dims['height'],
            'inner_width': inner_dims['width'],
            'inner_height': inner_dims['height']
        }

    def analyze_sequence_stability(self, threshold_position: float = 5.0,
                                   threshold_angle: float = 5.0,
                                   frame_range: tuple = None) -> str:
        """
        Analyze stability of markers across sequence.
        """
        df = self.data.df
        if frame_range:
            start_frame, end_frame = frame_range
            df = df[(df['frame_index'] >= start_frame) & (df['frame_index'] <= end_frame)]

        df = df.sort_values('frame_index')
        anomalies = []

        # Get initial values
        initial_centers = self._get_marker_centers(df.iloc[0])
        initial_angles = {i: df.iloc[0][f'id{i}_angle'] for i in range(4)}

        # Analyze each frame
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            frame_index = current_row['frame_index']
            current_centers = self._get_marker_centers(current_row)

            for marker_id in range(4):
                # Position change
                position_change = self._calculate_distance(
                    initial_centers[marker_id],
                    current_centers[marker_id]
                )

                # Angle change
                angle_change = abs(current_row[f'id{marker_id}_angle'] - initial_angles[marker_id])

                if position_change > threshold_position:
                    anomalies.append(f"{frame_index}:{marker_id}[pos:{position_change:.1f}px]")
                if angle_change > threshold_angle:
                    anomalies.append(f"{frame_index}:{marker_id}[ang:{angle_change:.1f}°]")

        if anomalies:
            return f"Stability anomalies detected in frame:marker -> {', '.join(anomalies)}"
        return "No stability anomalies detected"

    def analyze_geometric_consistency(self, shape_tolerance: float = 0.02,
                                      frame_range: tuple = None) -> str:
        """
        Analyze geometric consistency by comparing ratios to the first frame.

        Args:
            shape_tolerance: Maximum allowed deviation in ratios (2%)
            frame_range: Optional tuple (start_frame, end_frame) to analyze specific range

        Returns:
            str: Report with detected geometric anomalies
        """
        df = self.data.df
        if frame_range:
            start_frame, end_frame = frame_range
            df = df[(df['frame_index'] >= start_frame) & (df['frame_index'] <= end_frame)]

        df = df.sort_values('frame_index')
        anomalies = []

        # Get reference ratios from first frame only
        reference_ratios = self.calculate_aspect_ratio(df.iloc[0]['frame_index'])
        if not reference_ratios:
            return "Could not calculate reference aspect ratios"

        # Compare each frame with reference frame
        for i in range(1, len(df)):  # start from second frame
            curr_frame = df.iloc[i]['frame_index']
            curr_ratios = self.calculate_aspect_ratio(curr_frame)

            if not curr_ratios:
                continue

            # Check changes in all ratio types
            for ratio_type in ['inner_ratio', 'outer_ratio', 'average_ratio']:
                change = abs(curr_ratios[ratio_type] - reference_ratios[ratio_type]) / reference_ratios[ratio_type]
                if change > shape_tolerance:
                    anomalies.append(f"{curr_frame}[{ratio_type}:{change:.3f}]")

        if anomalies:
            return f"Geometric anomalies detected -> {', '.join(anomalies)}"
        return "No geometric anomalies detected"