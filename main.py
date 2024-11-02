from data import Data
from data_processor import DataProcessor
from detector import ArucoDetector
from image import Image
from export_data import ExportData


def process_all_frames(image_dir: str, expected_markers: int = 4, initial_contrast: int = 0):
    """
    Process all frames in directory using ArucoDetector

    Args:
        image_dir: Directory containing image frames
        expected_markers: Number of markers to detect (default: 4)
        initial_contrast: Initial contrast value for detection (default: 0)

    Returns:
        Data object with detection results
    """
    # Initialize objects
    image_loader = Image(image_dir)
    data = Data(expected_markers)
    detector = ArucoDetector(expected_markers=expected_markers,
                             initial_contrast=initial_contrast)

    total_frames = image_loader.get_total_frames()
    total_frames = 4
    print(f"Processing {total_frames} frames...")

    # Process each frame
    for idx in range(total_frames):
        frame, frame_number = image_loader.get_frame_by_index(idx)

        if frame is None:
            print(f"Error: Could not load frame at index {idx}")
            continue

        print(f"\nProcessing frame {frame_number} ({idx + 1}/{total_frames})")

        # Update initial contrast based on previous frame results
        if idx > 0:
            detector.initial_contrast = detector.get_initial_contrast(data, frame_number)

        # Detect markers
        success = detector.detect(frame, data, frame_number)

        if not success:
            print(f"Warning: No markers detected in frame {frame_number}")
            continue

        # Print detection summary for this frame
        detected_count = len(detector.detected_markers)
        print(f"Successfully detected {detected_count}/{expected_markers} markers")

    print("\nProcessing complete!")
    print(f"Total frames processed: {total_frames}")

    return data


# Example usage:
if __name__ == "__main__":
    # Directory containing image frames
    image_directory = "img/aruco4"

    # Process all frames
    result_data = process_all_frames(
        image_dir=image_directory,
        expected_markers=4,
        initial_contrast=0
    )

    # Now you can use the result_data object for cornerpin export or other operations
    exporter = ExportData(result_data)
    exporter.export_cornerpin("export/nuke_output.nk", point_type='outer')