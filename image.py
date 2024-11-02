import os
import re
import cv2


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

        Args:
            filename: Name of the image file

        Returns:
            int: Frame index if found, None otherwise
        """
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else None

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