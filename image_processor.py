import cv2 as cv
import numpy as np


class ImageProcessor:
    """
    Class for different image processing methods.
    """

    @staticmethod
    def enhance_contrast_legacy(image, contrast):
        """
        Enhance contrast using Photoshop-like legacy formula.

        Args:
            image: Input grayscale image
            contrast: Contrast value in range [-100, 100]
        """
        f = image.astype(float)
        contrast_factor = contrast / 100.0
        adjusted = (f - 128) * (1 + contrast_factor) + 128
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def enhance_contrast_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Enhance contrast using CLAHE method.
        """
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    @staticmethod
    def enhance_contrast_equalizer(image):
        """
        Enhance contrast using histogram equalization.
        """
        return cv.equalizeHist(image)