import cv2 as cv
import numpy as np


class ArucoDetector:
    def __init__(self, expected_markers=4):
        self.dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
        self.expected_markers = expected_markers

        # Limity parametrów
        self.win_size_range = range(51, 29, -2)
        self.thresh_constant_range = range(3, 13, 2)
        self.min_perimeter_range = [0.01, 0.03, 0.05]
        self.approx_accuracy_range = [0.01, 0.03, 0.05]
        self.corner_distance_range = [0.03, 0.05, 0.07]

    def calculate_missing_corner(self, corners, ids):
        if len(corners) != 3:
            return None, None

        # Znajdź brakujące ID
        detected_ids = ids.flatten()
        missing_id = None
        for i in range(4):
            if i not in detected_ids:
                missing_id = i
                break

        # Konwertuj corners na punkty (x,y)
        points_dict = {}  # słownik id -> punkt
        for i, corner in enumerate(corners):
            x = corner[0][:, 0].mean()
            y = corner[0][:, 1].mean()
            points_dict[detected_ids[i]] = (x, y)

        # Znajdź punkt przeciwległy do brakującego
        opposite_id = (missing_id + 2) % 4
        if opposite_id not in points_dict:
            print("Nie znaleziono przeciwległego punktu!")
            return None, None

        # Środek prostokąta to środek między dowolną parą przeciwległych punktów
        p_opposite = points_dict[opposite_id]

        # Obliczamy środek prostokąta jako średnią wszystkich punktów
        center_x = sum(p[0] for p in points_dict.values()) / 3
        center_y = sum(p[1] for p in points_dict.values()) / 3

        # Brakujący punkt jest symetryczny do przeciwległego względem środka
        missing_x = 2 * center_x - p_opposite[0]
        missing_y = 2 * center_y - p_opposite[1]

        print(f"Obliczenia dla markera {missing_id}:")
        print(f"Punkt przeciwległy (id={opposite_id}): ({p_opposite[0]:.2f}, {p_opposite[1]:.2f})")
        print(f"Środek: ({center_x:.2f}, {center_y:.2f})")
        print(f"Obliczony brakujący punkt: ({missing_x:.2f}, {missing_y:.2f})")

        # Tworzymy cztery rogi markera
        marker_size = 50  # aproksymowany rozmiar markera
        missing_corner = np.array([[
            [missing_x - marker_size / 2, missing_y - marker_size / 2],
            [missing_x + marker_size / 2, missing_y - marker_size / 2],
            [missing_x + marker_size / 2, missing_y + marker_size / 2],
            [missing_x - marker_size / 2, missing_y + marker_size / 2]
        ]], dtype=np.float32)

        return missing_corner, np.array([missing_id])

    def detect_single_marker(self, gray, marker_id, params):
        detector = cv.aruco.ArucoDetector(self.dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            for i, id in enumerate(ids):
                if id[0] == marker_id:
                    return corners[i], params
        return None, None

    def create_params(self, win_size, thresh_const, min_perim, approx_acc, corner_dist):
        params = cv.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = win_size
        params.adaptiveThreshWinSizeMax = win_size
        params.adaptiveThreshConstant = thresh_const
        params.minMarkerPerimeterRate = min_perim
        params.polygonalApproxAccuracyRate = approx_acc
        params.minCornerDistanceRate = corner_dist
        return params

    def detect(self, frame, auto=False):
        if frame is None:
            return None, None

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        default_params = self.create_params(
            win_size=51,
            thresh_const=7,
            min_perim=0.03,
            approx_acc=0.03,
            corner_dist=0.05
        )

        detector = cv.aruco.ArucoDetector(self.dictionary, default_params)
        corners, ids, _ = detector.detectMarkers(gray)

        all_corners = []
        all_ids = []

        if ids is not None:
            detected_ids = ids.flatten()
            for id in range(self.expected_markers):
                if id in detected_ids:
                    idx = np.where(detected_ids == id)[0][0]
                    all_corners.append(corners[idx])
                    all_ids.append(np.array([id], dtype=np.int32))  # Zmiana formatu
                else:
                    print(f"\nTrying to detect marker {id}...")
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
                                            all_corners.append(corner)
                                            all_ids.append(np.array([id], dtype=np.int32))  # Zmiana formatu
                                            print(f"Found marker {id} with parameters:")
                                            print(f"  Window size: {win_size}")
                                            print(f"  Threshold constant: {thresh_const}")
                                            print(f"  Min perimeter rate: {min_perim}")
                                            print(f"  Approx accuracy: {approx_acc}")
                                            print(f"  Corner distance: {corner_dist}")
                                            marker_found = True
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

        # Jeśli wykryto dokładnie 3 markery, oblicz pozycję czwartego
        if len(all_corners) == 3:
            print("\nDetected exactly 3 markers, calculating the position of the missing one...")
            missing_corner, missing_id = self.calculate_missing_corner(all_corners, np.array(all_ids))
            if missing_corner is not None and missing_id is not None:
                all_corners.append(missing_corner)
                all_ids.append(np.array(missing_id, dtype=np.int32))  # Zmiana formatu

        if all_corners and all_ids:
            # Debug print
            print("\nID formats before conversion:")
            for id_val in all_ids:
                print(
                    f"ID type: {type(id_val)}, shape: {id_val.shape if hasattr(id_val, 'shape') else 'no shape'}, value: {id_val}")

            ids_array = np.vstack(all_ids)  # Zmiana sposobu konwersji
            return all_corners, ids_array
        return None, None
