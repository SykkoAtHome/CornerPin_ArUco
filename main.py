from data import Data
from data_processor import DataProcessor
from detector import ArucoDetector
from image import Image
from export_data import ExportData


def process_all_frames(image_dir: str, expected_markers: int = 4):
    """
    Process all frames in directory using ArucoDetector.
    Each frame is processed independently.
    """
    image_loader = Image(image_dir)
    data = Data(expected_markers)
    detector = ArucoDetector(expected_markers=expected_markers)

    total_frames = image_loader.get_total_frames()
    # total_frames = 1
    print(f"Processing {total_frames} frames...")

    # Process each frame
    for idx in range(total_frames):
        frame, frame_number, file_path = image_loader.get_frame_by_index(idx)

        if frame is None:
            print(f"Error: Could not load frame at index {idx}")
            continue

        print(f"\nProcessing frame {frame_number} ({idx + 1}/{total_frames})")
        print(f"File: {file_path}")

        # Przekazujemy file_path do metody detect
        success = detector.detect(frame, data, frame_number, file_path)

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
        expected_markers=4
    )

    # Now you can use the result_data object for cornerpin export or other operations

    data_processor = DataProcessor(result_data)
    stability_report = data_processor.analyze_sequence_stability(threshold_position=5, threshold_angle=3)
    print(stability_report)

    geometry_report = data_processor.analyze_geometric_consistency()
    print(geometry_report)

    exporter = ExportData(result_data)
    exporter.export_cornerpin("export/aruco4.nk", point_type='outer')
    print(result_data.df.to_string())
