from dataclasses import dataclass
from typing import List, Tuple
import os
import cv2
import numpy as np
import pandas as pd

from visualization_config import VisualizationConfig, TextConfig


@dataclass
class MarkerElements:
    outline: bool = True
    corners: bool = True
    corner_numbers: bool = True
    center: bool = True
    id: bool = True
    orientation: bool = True
    angle: bool = True


class Draw:
    def __init__(self):
        """
        Initialize Draw class with default configuration.
        """
        self.config = VisualizationConfig()
        self.visible_markers = [True] * 4
        self.marker_elements = MarkerElements()

    def set_marker_elements(self, elements: MarkerElements) -> None:
        """
        Set which elements of markers should be visible.

        Args:
            elements: MarkerElements dataclass instance with boolean flags
        """
        self.marker_elements = elements

    def set_marker_visibility(self, marker_ids: List[int]) -> None:
        """
        Set which markers should be visible.

        Args:
            marker_ids: List of marker IDs that should be visible (0-3)
        """
        self.visible_markers = [False] * 4
        for marker_id in marker_ids:
            if 0 <= marker_id < 4:
                self.visible_markers[marker_id] = True

    def _draw_text_with_background(self, img: np.ndarray, text: str,
                                   position: Tuple[int, int],
                                   text_config: TextConfig) -> None:
        """
        Draw text with background and alpha blending.

        Args:
            img: Image to draw on (BGRA)
            text: Text to draw
            position: Position (x, y) to draw text
            text_config: TextConfig instance with text parameters
        """
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX,
            text_config.font_scale, text_config.thickness
        )

        x, y = position
        bg_rect = [
            (x - text_config.padding, y - text_height - text_config.padding),
            (x + text_width + text_config.padding, y + text_config.padding)
        ]

        # Convert background color to BGRA
        bg_color = (*text_config.bg_color, int(text_config.bg_alpha * 255))
        cv2.rectangle(img, bg_rect[0], bg_rect[1], bg_color, -1)

        # Convert text color to BGRA (full opacity)
        text_color = (*text_config.text_color, 255)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    text_config.font_scale, text_color,
                    text_config.thickness)

    def _draw_marker(self, img: np.ndarray, corners: np.ndarray,
                     marker_id: int, angle: float) -> None:
        """
        Draw single marker with selected elements.

        Args:
            img: Image to draw on (BGRA)
            corners: Corner points of the marker
            marker_id: ID of the marker
            angle: Orientation angle of the marker
        """
        # Convert corners to integer points
        corners = corners.astype(np.int32)

        # Draw marker outline
        if self.marker_elements.outline:
            pts = corners.reshape((-1, 1, 2))
            outline_color = (*self.config.marker_outline_color, 255)  # BGRA
            cv2.polylines(img, [pts], True,
                          outline_color,
                          self.config.marker_outline_thickness)

        # Draw corner points and numbers
        if self.marker_elements.corners or self.marker_elements.corner_numbers:
            for idx, point in enumerate(corners):
                pt = tuple(point)
                if self.marker_elements.corners:
                    corner_color = (*self.config.corner_colors[idx], 255)  # BGRA
                    cv2.circle(img, pt, 4, corner_color, -1)

                if self.marker_elements.corner_numbers:
                    text_pos = (pt[0] + self.config.corner.offset_x,
                                pt[1] + self.config.corner.offset_y)
                    self._draw_text_with_background(img, str(idx),
                                                    text_pos, self.config.corner)

        # Calculate and draw center point
        if self.marker_elements.center or self.marker_elements.id:
            center = corners.mean(axis=0).astype(int)

            if self.marker_elements.center:
                center_color = (*self.config.center_point_color, 255)  # BGRA
                cv2.circle(img, tuple(center),
                           self.config.center_point_size,
                           center_color, -1)

            if self.marker_elements.id:
                id_pos = (center[0] + self.config.marker_id.offset_x,
                          center[1] + self.config.marker_id.offset_y)
                self._draw_text_with_background(img, f'ID: {marker_id}',
                                                id_pos, self.config.marker_id)

        # Draw orientation and angle
        if not np.isnan(angle):
            if self.marker_elements.orientation:
                first_corner = tuple(corners[0].astype(int))
                second_corner = tuple(corners[1].astype(int))
                arrow_color = (*self.config.orientation_arrow_color, 255)  # BGRA
                cv2.arrowedLine(img, first_corner, second_corner,
                                arrow_color,
                                self.config.orientation_arrow_thickness)

            if self.marker_elements.angle:
                center = corners.mean(axis=0).astype(int)
                angle_pos = (center[0] + self.config.angle.offset_x,
                             center[1] + self.config.angle.offset_y)
                self._draw_text_with_background(img, f'{angle:.1f}°',
                                                angle_pos, self.config.angle)

    def draw_relative_center(self, img: np.ndarray,
                             center_x: float, center_y: float) -> None:
        """
        Draw the center point of the rectangle formed by markers.

        Args:
            img: Image to draw on (BGRA)
            center_x: X coordinate of the rectangle center
            center_y: Y coordinate of the rectangle center
        """
        center_point = (int(center_x), int(center_y))
        center_color = (*self.config.rect_center_color, 255)  # BGRA

        # Draw main center point
        cv2.circle(img, center_point,
                   self.config.rect_center_point_size * 2,
                   center_color, -1)

        # Draw crosshair
        line_length = self.config.rect_center_line_length
        cv2.line(img,
                 (center_point[0] - line_length, center_point[1]),
                 (center_point[0] + line_length, center_point[1]),
                 center_color,
                 self.config.rect_center_line_thickness)

        cv2.line(img,
                 (center_point[0], center_point[1] - line_length),
                 (center_point[0], center_point[1] + line_length),
                 center_color,
                 self.config.rect_center_line_thickness)

        # Draw coordinates text
        text = f"Center: ({center_x:.1f}, {center_y:.1f})"
        text_pos = (center_point[0] + self.config.rect_center_text.offset_x,
                    center_point[1] + self.config.rect_center_text.offset_y)
        self._draw_text_with_background(img, text, text_pos,
                                        self.config.rect_center_text)

    def export_as_image(self, data, output_dir: str) -> None:
        """
        Export marker detections as transparent PNG images.

        Args:
            data: Data object or DataFrame containing detection data
            output_dir: Directory where images will be saved
        """
        # Convert Data object to DataFrame if needed
        df = data.df if hasattr(data, 'df') else data
        df = df.sort_values('frame_index')

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        for _, row in df.iterrows():
            frame_index = int(row['frame_index'])
            width = int(row['image_width'])
            height = int(row['image_height'])

            # Create transparent image (BGRA)
            img_markers = np.zeros((height, width, 4), dtype=np.uint8)

            # Draw only visible markers
            for marker_id in range(4):
                if not self.visible_markers[marker_id]:
                    continue

                corners = self._get_corners_from_row(row, marker_id)
                if corners is not None:
                    angle = row[f'id{marker_id}_angle']
                    self._draw_marker(img_markers, corners[0], marker_id, angle)

            # Draw relative center if available
            center_x = row['center_x']
            center_y = row['center_y']

            if not (np.isnan(center_x) or np.isnan(center_y)):
                self.draw_relative_center(img_markers, center_x, center_y)

            # Save image
            filename = f"detection_{frame_index:04d}.png"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img_markers)

    def _get_corners_from_row(self, row: pd.Series, marker_id: int) -> np.ndarray:
        """
        Extract marker corners from DataFrame row.

        Args:
            row: DataFrame row containing detection data
            marker_id: ID of the marker

        Returns:
            numpy array in (1,4,2) format or None if corners not found
        """
        corners = np.zeros((4, 2))
        for i in range(4):
            x = row[f'id{marker_id}_corner{i}_x']
            y = row[f'id{marker_id}_corner{i}_y']
            if np.isnan(x) or np.isnan(y):
                return None
            corners[i, 0] = x
            corners[i, 1] = y

        # Transform to (1,4,2) format required by OpenCV
        return corners.reshape(1, 4, 2)