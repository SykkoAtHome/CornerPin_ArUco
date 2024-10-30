import cv2
from image import Image
from detector import ArucoDetector
from data import Data

# Inicjalizacja
EXPECTED_MARKERS = 4
# img_loader = Image('img/aruco2/')
img_loader = Image('img/aruco_errors/')
detector = ArucoDetector(expected_markers=EXPECTED_MARKERS)
data = Data(expected_markers=EXPECTED_MARKERS)

# Test na jednym obrazie
frame, index = img_loader.get_frame_by_index(1)
corners, ids = detector.detect(frame, auto=True)

if corners is not None and ids is not None:
    # Przygotowanie danych do zapisu
    markers_data = []
    for i, corner in enumerate(corners):
        x = corner[0][:, 0].mean()
        y = corner[0][:, 1].mean()
        markers_data.append((ids[i][0], x, y))

    # Dodanie danych do DataFrama
    data.add_detection(index, markers_data)

    # Wizualizacja
    img_markers = frame.copy()
    cv2.aruco.drawDetectedMarkers(img_markers, corners, ids)
    cv2.imshow(f'Frame {index} with markers', img_markers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(data.df)