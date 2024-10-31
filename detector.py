import cv2 as cv
import numpy as np


class ArucoDetector:
    """
    ArUco marker detector configured for a system with 4 markers forming a rectangle.

    Marker arrangement convention:
    - Markers form a rectangle in the image
    - Each marker follows ArUco corner ordering:
      0: top-left, 1: top-right, 2: bottom-right, 3: bottom-left

    Marker IDs convention:
    - ID0: top-left corner of the rectangle
    - ID1: top-right corner of the rectangle
    - ID2: bottom-left corner of the rectangle
    - ID3: bottom-right corner of the rectangle

    Each marker should be oriented so that its local corner ordering (0,1,2,3)
    matches the global rectangle orientation. This ensures consistent
    angle measurements and geometric calculations.
    """

    def __init__(self, expected_markers=4):
        self.dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
        self.expected_markers = expected_markers

        # Parameter ranges for detection
        self.win_size_range = range(51, 29, -2)
        self.thresh_constant_range = range(3, 13, 2)
        self.min_perimeter_range = [0.01, 0.03, 0.05]
        self.approx_accuracy_range = [0.01, 0.03, 0.05]
        self.corner_distance_range = [0.03, 0.05, 0.07]

    def detect_single_marker(self, gray, marker_id, params):
        """
        Attempts to detect a single marker with given ID.
        Returns tuple (corners, params) or (None, None).
        """
        detector = cv.aruco.ArucoDetector(self.dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            for i, id in enumerate(ids):
                if id[0] == marker_id:
                    return corners[i], params
        return None, None

    def create_params(self, win_size, thresh_const, min_perim, approx_acc, corner_dist):
        """Creates detection parameters with given values."""
        params = cv.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = win_size
        params.adaptiveThreshWinSizeMax = win_size
        params.adaptiveThreshConstant = thresh_const
        params.minMarkerPerimeterRate = min_perim
        params.polygonalApproxAccuracyRate = approx_acc
        params.minCornerDistanceRate = corner_dist
        return params

    def detect(self, frame, data, frame_index):
        """
        Detects markers and adds the data directly to DataFrame.

        Args:
            frame: Input image frame
            data: Data class instance to store results
            frame_index: Frame number

        Returns:
            bool: True if any markers were detected
        """
        if frame is None:
            return False

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Initialize row data
        row_data = {
            'frame_index': frame_index,
            'image_width': frame.shape[1],
            'image_height': frame.shape[0]
        }

        # Initialize all marker values as NaN
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

        # First attempt with default parameters
        default_params = self.create_params(
            win_size=51,
            thresh_const=7,
            min_perim=0.03,
            approx_acc=0.03,
            corner_dist=0.05
        )

        detector = cv.aruco.ArucoDetector(self.dictionary, default_params)
        corners, ids, _ = detector.detectMarkers(gray)

        markers_detected = False
        if ids is not None:
            detected_ids = ids.flatten()
            for id in range(self.expected_markers):
                if id in detected_ids:
                    idx = np.where(detected_ids == id)[0][0]
                    self._add_marker_data(row_data, id, corners[idx], default_params)
                    print(f"Marker {id} detected with default parameters")
                    markers_detected = True
                else:
                    print(f"\nTrying to detect marker {id}...")
                    # Try different parameters for missing marker
                    marker_found = False
                    for win_size in self.win_size_range:
                        for thresh_const in self.thresh_constant_range:
                            for min_perim in self.min_perimeter_range:
                                for approx_acc in self.approx_accuracy_range:
                                    for corner_dist in self.corner_distance_range:
                                        params = self.create_params(
                                            win_size, thresh_const, min_perim,
                                            approx_acc, corner_dist
                                        )
                                        corner, found_params = self.detect_single_marker(gray, id, params)

                                        if corner is not None:
                                            self._add_marker_data(row_data, id, corner, found_params)
                                            print(f"Found marker {id} with parameters:")
                                            print(f"  Window size: {win_size}")
                                            print(f"  Threshold constant: {thresh_const}")
                                            print(f"  Min perimeter rate: {min_perim}")
                                            print(f"  Approx accuracy: {approx_acc}")
                                            print(f"  Corner distance: {corner_dist}")
                                            marker_found = True
                                            markers_detected = True
                                            break
                                    if marker_found:
                                        break
                                if marker_found:
                                    break
                            if marker_found:
                                break
                        if marker_found:
                            break

                    if not marker_found:
                        print(f"Marker {id} not detected with any parameter combination")

        # Add row to DataFrame
        data.df.loc[len(data.df)] = row_data

        return markers_detected

    def _add_marker_data(self, row_data, marker_id, corners, params):
        """Helper method to add single marker data to row dictionary."""
        # Save corners
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

        # Save detection parameters
        row_data[f'id{marker_id}_win_size'] = params.adaptiveThreshWinSizeMin
        row_data[f'id{marker_id}_thresh_const'] = params.adaptiveThreshConstant
        row_data[f'id{marker_id}_min_perim'] = params.minMarkerPerimeterRate
        row_data[f'id{marker_id}_approx_acc'] = params.polygonalApproxAccuracyRate
        row_data[f'id{marker_id}_corner_dist'] = params.minCornerDistanceRate