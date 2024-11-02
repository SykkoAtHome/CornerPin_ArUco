import os
import re
import cv2
import numpy as np


class Image:
    def __init__(self, directory):
        """
        Initialize Image class for loading and managing frames.

        Args:
            directory: Path to directory with image frames
        """
        self.directory = directory
        self.images = []  # List of image paths
        self.frame_index = []  # List of frame indices
        self.load_images()

    def load_images(self):
        """
        Load image paths and their indices from directory.
        Sort them by frame index.
        """
        for filename in os.listdir(self.directory):
            file_path = os.path.join(self.directory, filename)
            if os.path.isfile(file_path):
                index = self.extract_index(filename)
                if index is not None:
                    self.images.append(file_path)
                    self.frame_index.append(index)

        # Sort images by frame index
        sorted_pairs = sorted(zip(self.frame_index, self.images))
        self.frame_index, self.images = zip(*sorted_pairs) if sorted_pairs else ([], [])
        self.frame_index = list(self.frame_index)
        self.images = list(self.images)

    def extract_index(self, filename):
        """
        Extract frame index from filename.
        Supported formats:
        - img_000.png -> 0
        - img.101.png -> 101
        - img009.png -> 9
        - img-050.png -> 50

        Args:
            filename: Name of the image file

        Returns:
            int: Frame index if found, None otherwise
        """
        # Look for number pattern that:
        # - can be preceded by _, ., - or nothing
        # - consists of digits (removing leading zeros)
        # - is followed by .png or other extension
        match = re.search(r'[_.-]?(\d+)\.[a-zA-Z]+$', filename)
        if match:
            # Convert to int to automatically remove leading zeros
            return int(match.group(1))
        return None

    def get_frame_by_index(self, index):
        """
        Load specific frame by its index in the sequence.

        Args:
            index: Index in the sequence (not frame number)

        Returns:
            tuple: (frame, frame_number) or (None, None) if index invalid
        """
        if 0 <= index < len(self.frame_index):
            frame_path = self.images[index]
            return cv2.imread(frame_path), self.frame_index[index]
        return None, None

    def get_total_frames(self):
        """
        Get total number of frames available.

        Returns:
            int: Number of frames
        """
        return len(self.frame_index)

    def save_image(self, image: np.ndarray, filename: str, directory: str = None) -> bool:
        """
        Save image to disk.

        Args:
            image: Image to save (can be BGR or BGRA)
            filename: Name of the output file
            directory: Directory to save the image (default: self.directory)

        Returns:
            bool: True if save successful, False otherwise
        """
        if directory is None:
            directory = self.directory

        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Full path for the output file
        output_path = os.path.join(directory, filename)

        try:
            # Save image
            return cv2.imwrite(output_path, image)
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return False