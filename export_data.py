from typing import Optional, Tuple
import numpy as np


class ExportData:
    """
    Class responsible for exporting tracking data to different formats
    """

    def __init__(self, data):
        """
        Initialize ExportData with Data object.

        Args:
            data: Data object containing markers information
        """
        self.data = data

    def _get_cornerpin_point(self, row, marker_id: int, point_type: str) -> Optional[Tuple[float, float]]:
        """
        Get point coordinates for cornerpin based on point type.
        Points are transformed to Nuke coordinate system where Y is inverted.

        Args:
            row: DataFrame row with marker data
            marker_id: ID of the marker
            point_type: Type of point to export ('center', 'outer', 'inner')

        Returns:
            Tuple of (x, y) coordinates or None if data invalid
        """
        try:
            # Get image height for Y coordinate transformation
            image_height = row['image_height']

            if point_type == 'center':
                x = row[f'id{marker_id}_center_x']
                y = row[f'id{marker_id}_center_y']
                if np.isnan(x) or np.isnan(y):
                    return None
                # Transform Y coordinate
                y = image_height - y
                return (x, y)

            # Mapping for outer/inner corners
            outer_corners = {
                0: (0, 0),  # ID0 corner 0
                1: (1, 1),  # ID1 corner 1
                2: (2, 2),  # ID2 corner 2
                3: (3, 3)  # ID3 corner 3
            }

            inner_corners = {
                0: (0, 2),  # ID0 corner 2
                1: (1, 3),  # ID1 corner 3
                2: (2, 0),  # ID2 corner 0
                3: (3, 1)  # ID3 corner 1
            }

            if point_type == 'outer':
                marker_id, corner_num = outer_corners[marker_id]
            else:  # inner
                marker_id, corner_num = inner_corners[marker_id]

            x = row[f'id{marker_id}_corner{corner_num}_x']
            y = row[f'id{marker_id}_corner{corner_num}_y']
            if np.isnan(x) or np.isnan(y):
                return None

            # Transform Y coordinate
            y = image_height - y
            return (x, y)

        except Exception as e:
            print(f"Error getting cornerpin point: {str(e)}")
            return None

    def export_cornerpin(self, output_path: str, point_type: str = 'center', frame_range: tuple = None) -> bool:
        """
        Export markers data to Nuke CornerPin2D format.

        Args:
            output_path: Path to output file
            point_type: Type of points to export ('center', 'outer', 'inner')
            frame_range: Tuple of (start_frame, end_frame) to export, None for all frames

        Returns:
            bool: True if export successful, False otherwise
        """
        # Validate point type
        if point_type not in ['center', 'outer', 'inner']:
            print(f"Invalid point type: {point_type}")
            return False

        # Get frame data
        df = self.data.df
        if frame_range:
            start_frame, end_frame = frame_range
            df = df[(df['frame_index'] >= start_frame) & (df['frame_index'] <= end_frame)]

        if df.empty:
            print("No data to export")
            return False

        try:
            with open(output_path, 'w') as f:
                f.write("CornerPin2D {\n")

                # Write points in correct order according to markers_info.txt
                # ID3 -> to1, ID2 -> to2, ID1 -> to3, ID0 -> to4
                marker_order = [3, 2, 1, 0]

                for i, marker_id in enumerate(marker_order):
                    points_x = []
                    points_y = []
                    last_valid_frame = None

                    for _, row in df.iterrows():
                        frame = int(row['frame_index'])
                        point = self._get_cornerpin_point(row, marker_id, point_type)

                        if point is None:
                            if last_valid_frame is not None:
                                points_x.append(f"x{frame}")
                                points_y.append(f"x{frame}")
                        else:
                            if last_valid_frame is None or frame > last_valid_frame + 1:
                                x_val = f"x{frame} {point[0]}"
                                y_val = f"x{frame} {point[1]}"
                            else:
                                x_val = str(point[0])
                                y_val = str(point[1])

                            points_x.append(x_val)
                            points_y.append(y_val)
                            last_valid_frame = frame

                    x_curve = " ".join(points_x)
                    y_curve = " ".join(points_y)

                    nuke_point = i + 1  # to1, to2, to3, to4
                    line = " to" + str(nuke_point) + " {{curve " + x_curve + "} {curve " + y_curve + "}}\n"
                    f.write(line)

                # Get actual image dimensions from DataFrame
                width = int(df['image_width'].iloc[0])
                height = int(df['image_height'].iloc[0])

                # Write 'from' points using actual image dimensions
                f.write(" from1 {0 0}\n")  # Top-left
                f.write(f" from2 {{{width} 0}}\n")  # Top-right
                f.write(f" from3 {{{width} {height}}}\n")  # Bottom-right
                f.write(f" from4 {{0 {height}}}\n")  # Bottom-left

                f.write(' invert true\n')
                f.write(' name CornerPin2D1\n')
                f.write(' xpos 0\n')
                f.write(' ypos 0\n')
                f.write('}\n')

            return True

        except Exception as e:
            print(f"Error exporting cornerpin: {str(e)}")
            return False