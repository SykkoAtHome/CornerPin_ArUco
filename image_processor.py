import cv2
import numpy as np

class ImageProcessor:
    """
    Class for image processing with legacy contrast enhancement method.
    """

    @staticmethod
    def enhance_contrast_legacy(image, contrast):
        """
        Enhance contrast using legacy formula.

        Args:
            image: Input BGR image (not grayscale)
            contrast: Contrast value in range [-100, 100]
        """
        f = image.astype(float)
        contrast_factor = contrast / 100.0
        adjusted = (f - 128) * (1 + contrast_factor) + 128
        return np.clip(adjusted, 0, 255).astype(np.uint8)
