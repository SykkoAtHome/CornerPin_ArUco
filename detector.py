import cv2
import cv2.aruco as aruco

def find_aruco_markers(image_path, marker_dict=aruco.DICT_4X4_50):
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load ArUco dictionary and parameters
    aruco_dict = aruco.Dictionary_get(marker_dict)
    aruco_params = aruco.DetectorParameters_create()

    # Detect markers
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        # Draw markers on the image
        img_with_markers = aruco.drawDetectedMarkers(img, corners, ids)
        cv2.imshow("Detected ArUco Markers", img_with_markers)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Extract positions
        positions = {int(id[0]): corner for id, corner in zip(ids, corners)}
        return positions
    else:
        print("No ArUco markers found.")
        return {}

# Usage:
image_path = "img/aruco1/aruco1_00000.png"
found_markers = find_aruco_markers(image_path)
print("Detected marker positions:", found_markers)
