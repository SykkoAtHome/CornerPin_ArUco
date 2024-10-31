import cv2 as cv
import numpy as np
from image_processor import ImageProcessor


class ArucoDetector:
    def __init__(self, expected_markers=4, contrast_step=15, steps=20, initial_contrast=0):
        self.expected_markers = expected_markers
        self.contrast_step = contrast_step
        self.steps = steps
        self.image_processor = ImageProcessor()
        self.initial_contrast = initial_contrast

        # Dictionary types to check
        self.dictionaries = {
            'DICT_4X4_50': cv.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv.aruco.DICT_4X4_100,
            'DICT_4X4_250': cv.aruco.DICT_4X4_250,
            'DICT_4X4_1000': cv.aruco.DICT_4X4_1000,
        }

        self.current_dict_name = None
        self.dictionary = None

        # Stage 1: Default parameters
        self.default_params = self.create_params(
            win_size=33,
            thresh_const=7,
            min_perim=0.03,
            approx_acc=0.03,
            corner_dist=0.05,
            min_marker_dist=0.1,
            perspective_remove_cell=4,
            max_erroneous_bits=0.35
        )

        # Stage 2: Quick scan parameters
        self.quick_params = self.create_params(
            win_size=23,
            thresh_const=5,
            min_perim=0.02,
            approx_acc=0.05,
            corner_dist=0.04,
            min_marker_dist=0.05,
            perspective_remove_cell=8,
            max_erroneous_bits=0.4
        )

        # Stage 3: Detailed search parameters ranges
        self.win_size_range = range(31, 41, 2)
        self.thresh_constant_range = range(5, 11, 2)
        self.min_perimeter_range = [0.02, 0.03, 0.04]

    def create_params(self, win_size, thresh_const, min_perim, approx_acc,
                      corner_dist, min_marker_dist, perspective_remove_cell,
                      max_erroneous_bits):
        params = cv.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = win_size
        params.adaptiveThreshWinSizeMax = win_size
        params.adaptiveThreshConstant = thresh_const
        params.minMarkerPerimeterRate = min_perim
        params.polygonalApproxAccuracyRate = approx_acc
        params.minCornerDistanceRate = corner_dist
        params.minMarkerDistanceRate = min_marker_dist
        params.perspectiveRemovePixelPerCell = perspective_remove_cell
        params.maxErroneousBitsInBorderRate = max_erroneous_bits
        params.minOtsuStdDev = 5.0
        params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        params.maxMarkerPerimeterRate = 4.0
        params.minDistanceToBorder = 3
        params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.1
        return params

    def process_image(self, image, contrast=0):
        """
        Process image with contrast enhancement.
        """
        if contrast == 0:
            return image

        return self.image_processor.enhance_contrast_legacy(image, contrast)

    def detect_dictionary(self, gray):
        if self.dictionary is None:
            best_detection = None
            max_markers = 0

            for dict_name, dict_id in self.dictionaries.items():
                dictionary = cv.aruco.getPredefinedDictionary(dict_id)
                detector = cv.aruco.ArucoDetector(dictionary, self.default_params)
                corners, ids, _ = detector.detectMarkers(gray)

                if ids is not None and len(ids) > max_markers:
                    max_markers = len(ids)
                    best_detection = (dict_name, dictionary)

                if max_markers >= self.expected_markers:
                    break

            if best_detection:
                self.current_dict_name = best_detection[0]
                self.dictionary = best_detection[1]
                return True
            return False
        return True

    def get_initial_contrast(self, data, current_frame_index):
        """
        Determine initial contrast based on previous frame data.
        Returns minimum contrast value from previous frame minus contrast_step.
        If no previous data available or all values are NaN, returns 0.
        """
        if current_frame_index <= 0:
            return 0

        # Get data from previous frame
        prev_frame = data.df[data.df['frame_index'] == current_frame_index - 1]
        if prev_frame.empty:
            return 0

        # Get contrast columns for all markers
        contrast_columns = [f'id{i}_contrast' for i in range(self.expected_markers)]
        contrast_values = prev_frame[contrast_columns].iloc[0]

        # Filter out NaN values and find minimum
        valid_contrasts = contrast_values.dropna()
        if valid_contrasts.empty:
            return 0

        # Calculate new initial contrast
        min_contrast = valid_contrasts.min()
        initial_contrast = min_contrast - self.contrast_step

        # Ensure contrast is not negative
        return max(0, initial_contrast)

    def detect(self, frame, data, frame_index):
        """
        Detect markers with cumulative contrast enhancement.
        Run all 3 stages for all markers on each contrast level to find best detection.
        """
        if frame is None:
            return False

        # Dictionary to store best detection for each marker
        best_detections = {}  # marker_id -> (corners, params, contrast_level)

        # Initial conversion to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Apply initial contrast if specified
        if self.initial_contrast > 0:
            gray = self.process_image(gray, self.initial_contrast)
            print(f"\n=== Starting with initial contrast: {self.initial_contrast} ===")

        # Main loop - try detection with increasing contrast
        for step in range(self.steps):
            current_contrast = self.initial_contrast + step * self.contrast_step
            print(f"\n=== Detection attempt {step + 1}/{self.steps} (contrast: {current_contrast}) ===")

            # Make sure we have correct dictionary
            if not self.detect_dictionary(gray):
                return False

            # Stage 1: Default Detection
            detector = cv.aruco.ArucoDetector(self.dictionary, self.default_params)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                for i, corners_i in enumerate(corners):
                    marker_id = int(ids[i][0])
                    if 0 <= marker_id < self.expected_markers:
                        best_detections[marker_id] = (corners_i, self.default_params, current_contrast)

            # Stage 2: Quick Scan
            detector = cv.aruco.ArucoDetector(self.dictionary, self.quick_params)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                for i, corners_i in enumerate(corners):
                    marker_id = int(ids[i][0])
                    if 0 <= marker_id < self.expected_markers:
                        best_detections[marker_id] = (corners_i, self.quick_params, current_contrast)

            # Stage 3: Detailed Search
            missing_ids = set(range(self.expected_markers)) - set(best_detections.keys())
            for marker_id in missing_ids:
                for win_size in self.win_size_range:
                    for thresh_const in self.thresh_constant_range:
                        for min_perim in self.min_perimeter_range:
                            params = self.create_params(
                                win_size=win_size,
                                thresh_const=thresh_const,
                                min_perim=min_perim,
                                approx_acc=0.03,
                                corner_dist=0.05,
                                min_marker_dist=0.1,
                                perspective_remove_cell=4,
                                max_erroneous_bits=0.35
                            )

                            detector = cv.aruco.ArucoDetector(self.dictionary, params)
                            corners, ids, _ = detector.detectMarkers(gray)

                            if ids is not None and marker_id in ids.flatten():
                                idx = np.where(ids.flatten() == marker_id)[0][0]
                                best_detections[marker_id] = (corners[idx], params, current_contrast)
                                break
                        if marker_id in best_detections:
                            break
                    if marker_id in best_detections:
                        break

            print(f"Found {len(best_detections)}/{self.expected_markers} markers at contrast {current_contrast}")

            # If we haven't found all markers, increase contrast for next iteration
            if len(best_detections) < self.expected_markers:
                gray = self.process_image(gray, self.contrast_step)
                continue

            # If we found all markers, we can still continue to look for better detections
            # at higher contrast levels

        # After all contrast levels, prepare final results
        self.detected_markers = []
        for marker_id, (corners, params, contrast) in best_detections.items():
            self.detected_markers.append((marker_id, corners, params, contrast))  # dodane contrast
            print(f"Marker {marker_id} best detection at contrast {contrast}")

        # Add final detections to data
        missing_ids = set(range(self.expected_markers)) - set(best_detections.keys())
        print(f"Detection complete. Found {len(best_detections)}/{self.expected_markers} markers. "
              f"Missing IDs: {sorted(missing_ids) if missing_ids else 'None'}")

        data.add_detection(frame_index, self.detected_markers, frame.shape[1], frame.shape[0])
        return len(self.detected_markers) > 0
