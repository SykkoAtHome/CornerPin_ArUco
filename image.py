import os
import re
import cv2
import numpy as np


class Image:
    def __init__(self, directory):
        self.directory = directory
        self.images = []
        self.frame_index = []
        self.load_images()

    def load_images(self):
        for filename in os.listdir(self.directory):
            file_path = os.path.join(self.directory, filename)
            if os.path.isfile(file_path):
                index = self.extract_index(filename)
                if index is not None:
                    self.images.append(file_path)
                    self.frame_index.append(index)
        # sortowanie po indeksach
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

    def draw_markers(self, frame, data, frame_index):
        """
        Draws markers on the frame based on DataFrame data.

        Args:
            frame: Input image frame
            data: Data class instance containing marker information
            frame_index: Index of the frame to process

        Returns:
            Image with drawn markers and additional visualization elements
        """
        img_markers = frame.copy()

        # Lists to store all markers' data for batch drawing
        all_corners = []
        all_ids = []

        for marker_id in range(data.expected_markers):
            # Get marker corners
            corners = data.get_marker_corners(frame_index, marker_id)
            if corners is not None:
                all_corners.append(corners)
                all_ids.append([marker_id])

                # Get inner array of points for additional visualization
                corners = corners[0]

                # 1. Draw corners as colored points with numbers
                # Red (0), Green (1), Blue (2), Yellow (3)
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
                for idx, (point, color) in enumerate(zip(corners, colors)):
                    pt = tuple(point.astype(int))
                    cv2.circle(img_markers, pt, 4, color, -1)
                    cv2.putText(img_markers, str(idx),
                                (pt[0] + 5, pt[1] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 2. Draw center point and marker ID
                center = corners.mean(axis=0).astype(int)
                cv2.circle(img_markers, tuple(center), 4, (255, 255, 255), -1)  # White center
                cv2.putText(img_markers, f'ID: {marker_id}',
                            (center[0] - 20, center[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # 3. Draw orientation angle
                angle = data.df.iloc[0][f'id{marker_id}_angle']
                if not np.isnan(angle):
                    # Draw arrow from corner 0 to corner 1
                    first_corner = tuple(corners[0].astype(int))
                    second_corner = tuple(corners[1].astype(int))
                    cv2.arrowedLine(img_markers, first_corner, second_corner,
                                    (255, 0, 0), 2)  # Blue arrow

                    # Draw angle value
                    cv2.putText(img_markers, f'{angle:.1f}°',
                                (center[0] - 40, center[1] + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 55, 25), 2)

        # Draw ArUco markers if any were detected
        if all_corners and all_ids:
            all_corners = np.array(all_corners, dtype=np.float32)
            all_ids = np.array(all_ids, dtype=np.int32)
            cv2.aruco.drawDetectedMarkers(img_markers, all_corners, all_ids)

        return img_markers