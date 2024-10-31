import os
import re
import cv2
import numpy as np
from visualization_config import VisualizationConfig


class Image:
    def __init__(self, directory):
        self.directory = directory
        self.images = []
        self.frame_index = []
        self.config = VisualizationConfig()
        self.load_images()

    def load_images(self):
        for filename in os.listdir(self.directory):
            file_path = os.path.join(self.directory, filename)
            if os.path.isfile(file_path):
                index = self.extract_index(filename)
                if index is not None:
                    self.images.append(file_path)
                    self.frame_index.append(index)
        sorted_pairs = sorted(zip(self.frame_index, self.images))
        self.frame_index, self.images = zip(*sorted_pairs) if sorted_pairs else ([], [])
        self.frame_index = list(self.frame_index)
        self.images = list(self.images)

    def extract_index(self, filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else None

    def get_frame_by_index(self, index):
        if 0 <= index < len(self.frame_index):
            frame_path = self.images[index]
            return cv2.imread(frame_path), self.frame_index[index]
        return None, None

    def _draw_text_with_background(self, img, text, position, text_config):
        """
        Helper method to draw text with background and alpha blending
        """
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX,
            text_config.font_scale, text_config.thickness
        )

        # Calculate background rectangle coordinates
        x, y = position
        bg_rect = [
            (x - text_config.padding, y - text_height - text_config.padding),
            (x + text_width + text_config.padding, y + text_config.padding)
        ]

        # Create background overlay with alpha
        overlay = img.copy()
        cv2.rectangle(overlay, bg_rect[0], bg_rect[1], text_config.bg_color, -1)

        # Apply alpha blending
        alpha = text_config.bg_alpha
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Draw text
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    text_config.font_scale, text_config.text_color,
                    text_config.thickness)

    def draw_markers(self, frame, data, frame_index):
        """
        Draws markers on the frame based on DataFrame data.
        """
        img_markers = frame.copy()

        for marker_id in range(data.expected_markers):
            corners = data.get_marker_corners(frame_index, marker_id)
            if corners is not None:
                corners = corners[0]

                # Draw marker outline
                pts = corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_markers, [pts], True,
                              self.config.marker_outline_color,
                              self.config.marker_outline_thickness)

                # Draw corner points and numbers
                for idx, point in enumerate(corners):
                    pt = tuple(point.astype(int))
                    # Draw corner point
                    cv2.circle(img_markers, pt, 4,
                               self.config.corner_colors[idx], -1)

                    # Draw corner number with background
                    text_pos = (pt[0] + self.config.corner.offset_x,
                                pt[1] + self.config.corner.offset_y)
                    self._draw_text_with_background(img_markers, str(idx),
                                                    text_pos, self.config.corner)

                # Draw center point and marker ID
                center = corners.mean(axis=0).astype(int)
                cv2.circle(img_markers, tuple(center),
                           self.config.center_point_size,
                           self.config.center_point_color, -1)

                # Draw marker ID
                id_pos = (center[0] + self.config.marker_id.offset_x,
                          center[1] + self.config.marker_id.offset_y)
                self._draw_text_with_background(img_markers, f'ID: {marker_id}',
                                                id_pos, self.config.marker_id)

                # Draw orientation angle
                angle = data.df[data.df['frame_index'] == frame_index].iloc[0][f'id{marker_id}_angle']
                if not np.isnan(angle):
                    # Draw orientation arrow
                    first_corner = tuple(corners[0].astype(int))
                    second_corner = tuple(corners[1].astype(int))
                    cv2.arrowedLine(img_markers, first_corner, second_corner,
                                    self.config.orientation_arrow_color,
                                    self.config.orientation_arrow_thickness)

                    # Draw angle value
                    angle_pos = (center[0] + self.config.angle.offset_x,
                                 center[1] + self.config.angle.offset_y)
                    self._draw_text_with_background(img_markers, f'{angle:.1f}°',
                                                    angle_pos, self.config.angle)

        return img_markers
