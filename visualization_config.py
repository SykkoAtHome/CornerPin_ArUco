class TextConfig:
    def __init__(self, font_scale=0.5, text_color=(255, 255, 255),
                 bg_color=(32, 32, 32), bg_alpha=0.7, thickness=1,
                 padding=5, offset_x=0, offset_y=0):
        self.font_scale = font_scale
        self.text_color = text_color
        self.bg_color = bg_color
        self.bg_alpha = bg_alpha  # 0.0 to 1.0
        self.thickness = thickness
        self.padding = padding
        self.offset_x = offset_x
        self.offset_y = offset_y

class VisualizationConfig:
    def __init__(self):
        # Configuration for marker ID text
        self.marker_id = TextConfig(
            font_scale=0.6,
            text_color=(0, 255, 255),  # Cyan
            bg_color=(0, 0, 0),
            bg_alpha=0.25,
            padding=5,
            offset_x=0,
            offset_y=-50,
            thickness=2
        )

        # Configuration for angle text
        self.angle = TextConfig(
            font_scale=0.4,
            text_color=(90, 90, 215),
            bg_color=(0, 0, 0),
            bg_alpha=0.1,
            padding=5,
            offset_x=40,
            offset_y=60
        )

        # Configuration for corner number text
        self.corner = TextConfig(
            font_scale=0.3,
            text_color=(255, 255, 255),  # White
            bg_color=(32, 32, 32),
            bg_alpha=0.3,
            padding=3,
            offset_x=5,
            offset_y=15
        )

        # Corner point colors (Red, Green, Blue, Yellow)
        self.corner_colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 0)
        ]

        # Other visualization settings
        self.marker_outline_color = (0, 255, 0)  # Green
        self.marker_outline_thickness = 1
        self.center_point_color = (0, 255, 255)  # White
        self.center_point_size = 4
        self.orientation_arrow_color = (255, 0, 0)  # Blue
        self.orientation_arrow_thickness = 1