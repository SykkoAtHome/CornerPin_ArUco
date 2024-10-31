import cv2
import numpy as np
from detector import ArucoDetector
from image import Image

def visualize_marker_corners(frame, corners, ids):
    """
    Wizualizuje narożniki markera z numeracją i kierunkiem.
    """
    img = frame.copy()
    if corners is not None:
        for i, corner in enumerate(corners):
            # Rysuj narożniki
            pts = corner[0].astype(np.int32)
            marker_id = ids[i][0]

            # Rysuj punkty i numeruj je
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # BGR: czerwony, zielony, niebieski, żółty
            for j, pt in enumerate(pts):
                cv2.circle(img, tuple(pt), 5, colors[j], -1)
                cv2.putText(img, str(j), tuple(pt + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[j], 2)

            # Rysuj strzałkę kierunku (od punktu 0 do 1)
            start_point = tuple(pts[0])
            end_point = tuple(pts[1])
            cv2.arrowedLine(img, start_point, end_point, (0, 255, 255), 2)

            # Dodaj ID markera
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(img, f'ID: {marker_id}', tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img


def test_marker_rotation(frame, detector):
    corners, ids = detector.detect(frame)

    if corners is not None and ids is not None:
        # Wyświetl informacje o narożnikach
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            print(f"\nMarker ID {marker_id}:")
            for j, point in enumerate(corner[0]):
                print(f"Narożnik {j}: x={point[0]:.1f}, y={point[1]:.1f}")

            # Oblicz orientację markera (kąt między punktami 0 i 1)
            pts = corner[0]
            vec = pts[1] - pts[0]  # wektor od punktu 0 do 1
            angle = np.degrees(np.arctan2(vec[1], vec[0]))
            print(f"Orientacja markera (stopnie): {angle:.1f}")

        # Wizualizacja
        img_markers = visualize_marker_corners(frame, corners, ids)
        cv2.imshow('Markers with corners', img_markers)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Użycie
if __name__ == "__main__":
    img_loader = Image('img/aruco_errors/')
    detector = ArucoDetector(expected_markers=4)

    frame, index = img_loader.get_frame_by_index(1)
    test_marker_rotation(frame, detector)