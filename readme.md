# ArUco Markers Detection System

## Overview
This project implements a robust ArUco marker detection system with advanced contrast enhancement capabilities. The system is designed to detect and track four ArUco markers arranged in a rectangular pattern, making it suitable for camera calibration, pose estimation, and alignment applications.

## Key Features
- Detection of 4 ArUco markers in predefined positions:
  - ID 0: Top-left corner
  - ID 1: Top-right corner
  - ID 2: Bottom-right corner
  - ID 3: Bottom-left corner
- Adaptive contrast enhancement for improved marker detection
- Multi-stage detection process:
  1. Default parameters detection
  2. Quick scan with optimized parameters
  3. Detailed search with parameter sweeping
- Comprehensive data collection and storage
- Visual feedback with marker outlines, corners, IDs, and orientation
- Export capabilities to Nuke (CornerPin2D format)
- Aspect ratio calculation for marker rectangle

## Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy
- Pandas

## Project Structure
```
├── data.py                # Data management and storage
├── data_processor.py      # Data analysis and calculations
├── detector.py            # ArUco marker detection implementation
├── draw.py               # Visualization utilities
├── export_data.py        # Data export functionality
├── image.py              # Image loading and handling
├── image_processor.py    # Image processing utilities
├── main.py              # Main application entry point
└── visualization_config.py # Visualization settings
```

## Installation
1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Detection
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
```

### Analysis and Export
```python
# Calculate aspect ratio
processor = DataProcessor(data)
ratios = processor.calculate_aspect_ratio(frame_index=0)
if ratios:
    print(f"Marker rectangle aspect ratio: {ratios['average_ratio']:.6f}")

# Export to Nuke
exporter = ExportData(data)
exporter.export_cornerpin("output/cornerpin.nk", point_type='center')
```

### Visualization
```python
from draw import Draw, MarkerElements

# Configure visualization
drawer = Draw()
elements = MarkerElements(
    outline=True,
    corners=True,
    corner_numbers=True,
    center=True,
    id=True,
    orientation=True,
    angle=True
)
drawer.set_marker_elements(elements)

# Draw markers
markers_visualization = drawer.draw_markers(data, frame_index=0)
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

## Data Analysis
The system provides various analysis capabilities:

### Aspect Ratio Calculation
- Calculates aspect ratio using both outer and inner marker points
- Provides average ratio for improved accuracy
- Helps validate marker detection accuracy
- Can be used for calibration purposes

### Relative Center Calculation
- Calculates the center point of the marker rectangle
- Uses intersection of diagonal lines
- Provides additional validation of marker positions

## Export Capabilities
The system can export tracking data to various formats:

### Nuke CornerPin2D
- Exports marker positions as CornerPin2D node
- Supports different point types:
  - Center points
  - Outer corners
  - Inner corners
- Handles missing frames and interpolation
- Maintains correct coordinate system transformation

## Data Collection
The system collects comprehensive data for each detected marker:
- Corner coordinates (x, y for each corner)
- Center position
- Orientation angle
- Detection parameters
- Contrast level
- Frame information

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
[Your License Information Here]