import os
import re
import cv2

class Image:
    def __init__(self, directory):
        self.directory = directory
        self.images = []
        self.frame_index = []
        self.load_images()

    def load_images(self):
        for filename in os.listdir(self.directory):
            file_path = os.path.join(self.directory, filename)
            if os.path.isfile(file_path):
                index = self.extract_index(filename)
                if index is not None:  # dodane sprawdzenie
                    self.images.append(file_path)
                    self.frame_index.append(index)
        # sortowanie po indeksach
        sorted_pairs = sorted(zip(self.frame_index, self.images))
        self.frame_index, self.images = zip(*sorted_pairs) if sorted_pairs else ([], [])
        self.frame_index = list(self.frame_index)
        self.images = list(self.images)

    def extract_index(self, filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else None

    def get_frame_by_index(self, index):
        if 0 <= index < len(self.frame_index):
            frame_path = self.images[index]
            return cv2.imread(frame_path), self.frame_index[index]
        return None, None