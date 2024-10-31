import cv2 as cv
import numpy as np
from image_processor import ImageProcessor


class ArucoDetector:
    def __init__(self, expected_markers=4, contrast_step=5, steps=10):
        self.expected_markers = expected_markers
        self.contrast_step = contrast_step
        self.steps = steps
        self.image_processor = ImageProcessor()

        # Wybór metody przetwarzania kontrastu
        self.contrast_method = "legacy"  # można zmienić na: "clahe", "equalizer"

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
        Process image with selected contrast enhancement method.

        Args:
            image: Input grayscale image
            contrast: Contrast value (interpretation depends on method)
        Returns:
            Processed image
        """
        if contrast == 0:
            return image

        if self.contrast_method == "legacy":
            return self.image_processor.enhance_contrast_legacy(image, contrast)
        elif self.contrast_method == "clahe":
            return self.image_processor.enhance_contrast_clahe(image, clip_limit=contrast)
        elif self.contrast_method == "equalizer":
            return self.image_processor.enhance_contrast_equalizer(image)
        else:
            raise ValueError(f"Unknown contrast method: {self.contrast_method}")

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

    def detect(self, frame, data, frame_index):
        if frame is None:
            return False

        original_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Initialize detection results
        self.detected_markers = []
        found_ids = set()

        # Try detection with increasing contrast
        for step in range(self.steps):
            contrast = step * self.contrast_step
            current_gray = self.process_image(original_gray, contrast)

            if not self.detect_dictionary(current_gray):
                return False

            if contrast == 0:
                print(f"\n=== Starting detection with original image ===")
            else:
                print(f"\n=== Starting detection with {self.contrast_method} contrast {contrast} ===")

            # Stage 1: Default Detection
            print("\n=== Stage 1: Default Detection ===")
            detector = cv.aruco.ArucoDetector(self.dictionary, self.default_params)
            corners, ids, _ = detector.detectMarkers(current_gray)

            stage1_found = []
            if ids is not None:
                for i, corners_i in enumerate(corners):
                    marker_id = int(ids[i][0])
                    if 0 <= marker_id < self.expected_markers and marker_id not in found_ids:
                        self.detected_markers.append((marker_id, corners_i, self.default_params))
                        stage1_found.append(marker_id)
                        found_ids.add(marker_id)

            print(
                f"Found {len(stage1_found)}/{self.expected_markers - len(found_ids - set(stage1_found))} markers (ID={', ID='.join(map(str, sorted(stage1_found)))})" if stage1_found else f"Found 0/{self.expected_markers - len(found_ids)} markers")

            # If all markers found, finish detection
            if len(found_ids) == self.expected_markers:
                print("\n=== Detection complete. Success ===")
                data.add_detection(frame_index, self.detected_markers, frame.shape[1], frame.shape[0])
                return True

            # Stage 2: Quick Scan
            missing_ids = sorted(set(range(self.expected_markers)) - found_ids)
            print(f"\n=== Stage 2: Quick Scan ===")
            print(f"Looking for ID={', ID='.join(map(str, missing_ids))}")

            detector = cv.aruco.ArucoDetector(self.dictionary, self.quick_params)
            corners, ids, _ = detector.detectMarkers(current_gray)

            stage2_found = []
            if ids is not None:
                for i, corners_i in enumerate(corners):
                    marker_id = int(ids[i][0])
                    if marker_id in missing_ids:
                        self.detected_markers.append((marker_id, corners_i, self.quick_params))
                        stage2_found.append(marker_id)
                        found_ids.add(marker_id)

            print(
                f"Found {len(stage2_found)}/{len(missing_ids)} markers (ID={', ID='.join(map(str, sorted(stage2_found)))})" if stage2_found else f"Found 0/{len(missing_ids)} markers")

            # If all markers found, finish detection
            if len(found_ids) == self.expected_markers:
                print("\n=== Detection complete. Success ===")
                data.add_detection(frame_index, self.detected_markers, frame.shape[1], frame.shape[0])
                return True

            # Stage 3: Detailed Search
            missing_ids = sorted(set(range(self.expected_markers)) - found_ids)
            print(f"\n=== Stage 3: Detailed Search ===")
            print(f"Looking for ID={', ID='.join(map(str, missing_ids))}")

            stage3_found = []
            for marker_id in missing_ids:
                marker_found = False
                for win_size in self.win_size_range:
                    if marker_found:
                        break
                    for thresh_const in self.thresh_constant_range:
                        if marker_found:
                            break
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
                            corners, ids, _ = detector.detectMarkers(current_gray)

                            if ids is not None:
                                for i, id_val in enumerate(ids):
                                    if int(id_val[0]) == marker_id:
                                        self.detected_markers.append((marker_id, corners[i], params))
                                        stage3_found.append(marker_id)
                                        found_ids.add(marker_id)
                                        marker_found = True
                                        break
                            if marker_found:
                                break

            print(
                f"Found {len(stage3_found)}/{len(missing_ids)} markers (ID={', ID='.join(map(str, sorted(stage3_found)))})" if stage3_found else f"Found 0/{len(missing_ids)} markers")

            # If all markers found, finish detection
            if len(found_ids) == self.expected_markers:
                print("\n=== Detection complete. Success ===")
                data.add_detection(frame_index, self.detected_markers, frame.shape[1], frame.shape[0])
                return True

        # If we get here, we've tried all contrast levels and still haven't found all markers
        missing_ids = sorted(set(range(self.expected_markers)) - found_ids)
        print(f"\n=== Detection complete. Missing ID={', ID='.join(map(str, missing_ids))} ===")
        data.add_detection(frame_index, self.detected_markers, frame.shape[1], frame.shape[0])
        return len(self.detected_markers) > 0