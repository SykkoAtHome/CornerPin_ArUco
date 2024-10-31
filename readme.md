# ArUco Markers Detection System

## Overview
This project implements a robust ArUco marker detection system with advanced contrast enhancement capabilities. The system is designed to detect and track four ArUco markers arranged in a rectangular pattern, making it suitable for camera calibration, pose estimation, and alignment applications.

## Key Features
- Detection of 4 ArUco markers in predefined positions:
  - ID 0: Top-left corner
  - ID 1: Top-right corner
  - ID 2: Bottom-left corner
  - ID 3: Bottom-right corner
- Adaptive contrast enhancement for improved marker detection
- Multi-stage detection process:
  1. Default parameters detection
  2. Quick scan with optimized parameters
  3. Detailed search with parameter sweeping
- Comprehensive data collection and storage
- Visual feedback with marker outlines, corners, IDs, and orientation

## Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy
- Pandas

## Project Structure
```
├── data.py               # Data management and storage
├── detector.py           # ArUco marker detection implementation
├── image.py             # Image loading and visualization
├── image_processor.py   # Image processing utilities
├── main.py             # Main application entry point
├── test.py             # Testing utilities
└── visualization_config.py # Visualization settings
```

## Installation
1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install required dependencies:
```bash
pip install opencv-python numpy pandas
```

## Usage
1. Basic usage with main.py:
```python
from image import Image
from detector import ArucoDetector
from data import Data

# Initialize components
img_loader = Image('path/to/images/')
detector = ArucoDetector(expected_markers=4)
data = Data(expected_markers=4)

# Process single frame
frame, index = img_loader.get_frame_by_index(0)
success = detector.detect(frame, data, index)

# Visualize results
if success:
    img_markers = img_loader.draw_markers(frame, data, index)
```

## Detection Process
The system employs a three-stage detection process with increasing complexity:

1. **Default Detection**
   - Uses standard ArUco detection parameters
   - Quick initial scan for markers

2. **Quick Scan**
   - Uses optimized parameters for faster detection
   - Lower threshold values for marker identification

3. **Detailed Search**
   - Comprehensive parameter sweep
   - Adaptive contrast enhancement
   - Multiple dictionary types support

## Data Collection
The system collects comprehensive data for each detected marker:
- Corner coordinates (x, y for each corner)
- Center position
- Orientation angle
- Detection parameters
- Contrast level
- Frame information

## Visualization
The visualization system provides:
- Marker outlines with unique colors
- Numbered corners
- Marker IDs
- Orientation arrows
- Center points
- Detection angles

## Configuration
Detection parameters can be adjusted in `detector.py`:
- Window size for adaptive thresholding
- Threshold constants
- Minimum marker perimeter
- Corner distance parameters
- Marker distance parameters

Visualization settings can be modified in `visualization_config.py`:
- Colors for markers, corners, and text
- Font sizes and styles
- Text positioning
- Line thicknesses

## Known Limitations
- System expects exactly 4 markers in rectangular arrangement
- Performance may vary with image quality and lighting conditions
- Markers must be from supported ArUco dictionary types

## License
[Add your license information here]
