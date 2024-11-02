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

    @staticmethod
    def resize_image(image, scale_factor=1.0):
        """
        Resize image using Lanczos interpolation.

        Args:
            image: Input image (can be BGR or BGRA)
            scale_factor: Scale factor for resizing
                         1.0 = original size
                         0.5 = half size
                         2.0 = double size, etc.

        Returns:
            Resized image or None if error occurs
        """
        if image is None or scale_factor <= 0:
            return None

        try:
            # Get new dimensions
            height, width = image.shape[:2]
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Resize using Lanczos interpolation
            resized = cv2.resize(image,
                                 (new_width, new_height),
                                 interpolation=cv2.INTER_LANCZOS4)

            return resized

        except Exception as e:
            print(f"Error resizing image: {str(e)}")
            return None

    @staticmethod
    def merge_images(background: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """
        Merge two images with alpha blending.
        The overlay image should have an alpha channel (BGRA).

        Args:
            background: Background image (BGR)
            overlay: Overlay image with alpha channel (BGRA)

        Returns:
            BGR image with overlay blended onto background
        """
        # Validate input images
        if background is None or overlay is None:
            return None

        # Ensure images have the same dimensions
        if background.shape[:2] != overlay.shape[:2]:
            print("Error: Images have different dimensions")
            return None

        try:
            # Convert background to BGRA if needed
            if background.shape[2] == 3:
                background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

            # Split the overlay image into color and alpha channels
            overlay_color = overlay[:, :, :3]
            overlay_alpha = overlay[:, :, 3:4] / 255.0

            # Calculate alpha-blended result
            result = background.copy()
            result[:, :, :3] = (1.0 - overlay_alpha) * background[:, :, :3] + \
                              overlay_alpha * overlay_color

            # Convert back to BGR
            return cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)

        except Exception as e:
            print(f"Error merging images: {str(e)}")
            return None
