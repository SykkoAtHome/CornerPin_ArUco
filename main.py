from data import Data
from data_processor import DataProcessor
from detector import ArucoDetector
from image import Image
from export_data import ExportData


def process_all_frames(image_dir: str, expected_markers: int = 4, frame_range: tuple = None):
    """
    Process frames in directory using ArucoDetector.
    Each frame is processed independently.

    Args:
        image_dir: Directory containing image frames
        expected_markers: Number of markers to detect (default: 4)
        frame_range: Optional tuple (start_frame, end_frame) to process specific range
    """
    image_loader = Image(image_dir)
    data = Data(expected_markers)
    detector = ArucoDetector(expected_markers=expected_markers)

    # Get all available frames and determine which to process
    if frame_range:
        start_frame, end_frame = frame_range
        frames_to_process = [
            idx for idx, frame_num in enumerate(image_loader.frame_index)
            if start_frame <= frame_num <= end_frame
        ]
        print(f"Processing frames from {start_frame} to {end_frame}")
    else:
        total_frames = image_loader.get_total_frames()
        frames_to_process = range(total_frames)
        print(f"Processing all {len(frames_to_process)} frames...")

    # Process selected frames
    for i, idx in enumerate(frames_to_process):
        frame, frame_number, file_path = image_loader.get_frame_by_index(idx)

        if frame is None:
            print(f"Error: Could not load frame at index {idx}")
            continue

        print(f"\nProcessing frame {frame_number} ({i + 1}/{len(frames_to_process)})")
        print(f"File: {file_path}")

        success = detector.detect(frame, data, frame_number, file_path)

        if not success:
            print(f"Warning: No markers detected in frame {frame_number}")
            continue

        detected_count = len(detector.detected_markers)
        print(f"Successfully detected {detected_count}/{expected_markers} markers")

    print("\nProcessing complete!")
    print(f"Total frames processed: {len(frames_to_process)}")

    return data


# Example usage:
if __name__ == "__main__":
    # Directory containing image frames
    image_directory = "img/aruco4"

    # Process specific range of frames (e.g., frames 100-200)
    result_data = process_all_frames(
        image_dir=image_directory,
        expected_markers=4,
        frame_range=(475, 510)  # Optional: process only frames 100-200
    )

    # Run analysis
    data_processor = DataProcessor(result_data)
    stability_report = data_processor.analyze_sequence_stability(threshold_position=5, threshold_angle=3)
    print(stability_report)

    geometry_report = data_processor.analyze_geometric_consistency()
    print(geometry_report)

    # Export results
    exporter = ExportData(result_data)
    exporter.export_cornerpin("export/aruco4.nk", point_type='outer')
    print(result_data.df.to_string())