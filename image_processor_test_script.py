import cv2
import numpy as np
from image import Image
from image_processor import ImageProcessor
import os


def add_text_box(image, text):
    """
    Add a text box with processing parameters in the bottom-left corner
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    padding = 10

    # Split text into lines and get sizes
    lines = text.split('\n')
    line_heights = []
    line_widths = []
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        line_heights.append(h)
        line_widths.append(w)

    # Calculate box size
    box_width = max(line_widths) + 2 * padding
    box_height = sum(line_heights) + padding * (len(lines) + 1)

    # Calculate box position (bottom-left corner)
    box_x = 0
    box_y = image.shape[0] - box_height

    # Draw semi-transparent black box
    overlay = image.copy()
    cv2.rectangle(overlay, (box_x, box_y),
                  (box_x + box_width, image.shape[0]),
                  (0, 0, 0), -1)
    alpha = 0.7
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Add text lines
    y = box_y + padding + line_heights[0]
    for i, line in enumerate(lines):
        cv2.putText(image, line, (padding, y), font, font_scale,
                    (255, 255, 255), thickness)
        if i < len(lines) - 1:
            y += line_heights[i + 1] + padding

    return image


def enhance_contrast_legacy_bgr(image, contrast):
    """
    Apply legacy contrast enhancement to BGR image.
    """
    # Split the image into channels
    b, g, r = cv2.split(image)

    # Process each channel
    b = enhance_contrast_channel(b, contrast)
    g = enhance_contrast_channel(g, contrast)
    r = enhance_contrast_channel(r, contrast)

    # Merge channels back
    return cv2.merge([b, g, r])


def enhance_contrast_channel(channel, contrast):
    """
    Enhance contrast for a single channel.
    """
    f = channel.astype(float)
    contrast_factor = contrast / 100.0
    adjusted = (f - 128) * (1 + contrast_factor) + 128
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def enhance_clahe_bgr(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to BGR image.
    """
    # Split the image into channels
    b, g, r = cv2.split(image)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)

    # Apply CLAHE to each channel
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)

    # Merge channels back
    return cv2.merge([b, g, r])


def enhance_equalizer_bgr(image):
    """
    Apply histogram equalization to BGR image.
    """
    # Split the image into channels
    b, g, r = cv2.split(image)

    # Apply histogram equalization to each channel
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)

    # Merge channels back
    return cv2.merge([b, g, r])


def process_and_save_images(frame, output_dir='processed_images'):
    """
    Process image with different methods and parameters and save each result
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save original image
    original_with_text = add_text_box(frame.copy(), "Original image\nNo processing")
    cv2.imwrite(os.path.join(output_dir, '00_original.png'), original_with_text)

    # Legacy method with different contrast values
    contrasts = [20, 40, 60, 80, 100]
    for contrast in contrasts:
        processed = enhance_contrast_legacy_bgr(frame, contrast)
        text = f"Legacy method\nContrast: {contrast}"
        result = add_text_box(processed.copy(), text)
        cv2.imwrite(os.path.join(output_dir, f'01_legacy_contrast_{contrast}.png'), result)

    # CLAHE method with different clip limits
    clip_limits = [1.0, 2.0, 3.0, 4.0, 5.0]
    for clip_limit in clip_limits:  # Usunięto enumerate
        processed = enhance_clahe_bgr(frame, clip_limit=clip_limit)
        text = f"CLAHE method\nClip limit: {clip_limit}"
        result = add_text_box(processed.copy(), text)
        cv2.imwrite(os.path.join(output_dir, f'02_clahe_clip_{clip_limit:.1f}.png'), result)

    # Equalizer method
    processed = enhance_equalizer_bgr(frame)
    text = "Equalizer method\nNo parameters"
    result = add_text_box(processed.copy(), text)
    cv2.imwrite(os.path.join(output_dir, '03_equalizer.png'), result)

    print(f"All processed images have been saved to the '{output_dir}' directory.")
    print("\nImages saved:")
    print("1. Original image (color)")
    print("2. Legacy method variations with different contrast values (color)")
    print("3. CLAHE method variations with different clip limits (color)")
    print("4. Equalizer method (color)")


if __name__ == "__main__":
    # Load test image
    img_loader = Image('img/fix_alignment')
    frame, index = img_loader.get_frame_by_index(6)  # możesz zmienić indeks na ten z problematycznym markerem

    process_and_save_images(frame)