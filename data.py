import pandas as pd
import numpy as np


class Data:
    def __init__(self, expected_markers):
        self.expected_markers = expected_markers

        # Creating columns for DataFrame
        columns = [
            'frame_index',
            'image_width',
            'image_height'
        ]

        # Columns for each marker
        for id in range(expected_markers):
            # Corner coordinates (4 points, each has x,y)
            for corner in range(4):
                columns.extend([
                    f'id{id}_corner{corner}_x',
                    f'id{id}_corner{corner}_y'
                ])
            # Marker center (for backward compatibility and quick access)
            columns.extend([f'id{id}_center_x', f'id{id}_center_y'])
            # Marker orientation angle (in degrees)
            columns.append(f'id{id}_angle')
            # Detection parameters
            columns.extend([
                f'id{id}_win_size',
                f'id{id}_thresh_const',
                f'id{id}_min_perim',
                f'id{id}_approx_acc',
                f'id{id}_corner_dist'
            ])
            # Add contrast column for each marker
            columns.append(f'id{id}_contrast')

        self.df = pd.DataFrame(columns=columns)

    def add_detection(self, frame_index, markers_data, image_width=None, image_height=None):
        """
        Adds detections to DataFrame.

        markers_data: list of tuples (id, corners, params, contrast), where:
            - id: marker identifier
            - corners: array of corners (1,4,2)
            - params: detection parameters dictionary or None for defaults
            - contrast: contrast level at which marker was detected
        frame_index: frame number
        image_width: width of the processed image
        image_height: height of the processed image
        """
        row_data = {
            'frame_index': frame_index,
            'image_width': image_width if image_width is not None else np.nan,
            'image_height': image_height if image_height is not None else np.nan
        }

        # Initialize all values as NaN
        for id in range(self.expected_markers):
            # Corners
            for corner in range(4):
                row_data[f'id{id}_corner{corner}_x'] = np.nan
                row_data[f'id{id}_corner{corner}_y'] = np.nan
            # Center and orientation
            row_data[f'id{id}_center_x'] = np.nan
            row_data[f'id{id}_center_y'] = np.nan
            row_data[f'id{id}_angle'] = np.nan
            # Parameters
            row_data[f'id{id}_win_size'] = np.nan
            row_data[f'id{id}_thresh_const'] = np.nan
            row_data[f'id{id}_min_perim'] = np.nan
            row_data[f'id{id}_approx_acc'] = np.nan
            row_data[f'id{id}_corner_dist'] = np.nan
            # Initialize contrast
            row_data[f'id{id}_contrast'] = np.nan

        # Fill detected markers
        for marker_id, corners, params, contrast in markers_data:
            if marker_id < self.expected_markers:
                # corners has shape (1,4,2), so we take corners[0]
                for corner_idx, point in enumerate(corners[0]):
                    row_data[f'id{marker_id}_corner{corner_idx}_x'] = point[0]
                    row_data[f'id{marker_id}_corner{corner_idx}_y'] = point[1]

                # Calculate and save center
                center = corners[0].mean(axis=0)
                row_data[f'id{marker_id}_center_x'] = center[0]
                row_data[f'id{marker_id}_center_y'] = center[1]

                # Calculate and save orientation angle
                vec = corners[0][1] - corners[0][0]  # vector from point 0 to 1
                angle = np.degrees(np.arctan2(vec[1], vec[0]))
                row_data[f'id{marker_id}_angle'] = angle

                # Save contrast level
                row_data[f'id{marker_id}_contrast'] = contrast

                # Save detection parameters
                if params is not None:
                    row_data[f'id{marker_id}_win_size'] = params.adaptiveThreshWinSizeMin
                    row_data[f'id{marker_id}_thresh_const'] = params.adaptiveThreshConstant
                    row_data[f'id{marker_id}_min_perim'] = params.minMarkerPerimeterRate
                    row_data[f'id{marker_id}_approx_acc'] = params.polygonalApproxAccuracyRate
                    row_data[f'id{marker_id}_corner_dist'] = params.minCornerDistanceRate

        self.df.loc[len(self.df)] = row_data

    def get_marker_corners(self, frame_idx, marker_id):
        """
        Returns corners of specific marker for given frame in (1,4,2) format.
        If marker not found or data incomplete, returns None.

        Args:
            frame_idx: Frame index to get corners for
            marker_id: Marker ID to get corners for

        Returns:
            numpy array in (1,4,2) format or None if corners not found
        """
        row = self.df[self.df['frame_index'] == frame_idx]
        if row.empty:
            return None

        row = row.iloc[0]
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

    def get_marker_params(self, frame_idx, marker_id):
        """
        Returns detection parameters for specific marker.
        """
        row = self.df[self.df['frame_index'] == frame_idx]
        if row.empty:
            return None

        row = row.iloc[0]
        params = {
            'win_size': row[f'id{marker_id}_win_size'],
            'thresh_const': row[f'id{marker_id}_thresh_const'],
            'min_perim': row[f'id{marker_id}_min_perim'],
            'approx_acc': row[f'id{marker_id}_approx_acc'],
            'corner_dist': row[f'id{marker_id}_corner_dist']
        }
        return params if not np.isnan(list(params.values())).any() else None