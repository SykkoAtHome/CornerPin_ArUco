import pandas as pd
import numpy as np


class Data:
    def __init__(self, expected_markers):
        self.expected_markers = expected_markers
        columns = []
        for id in range(expected_markers):
            columns.extend([f'id{id}_x', f'id{id}_y'])

        self.df = pd.DataFrame(columns=['frame_index'] + columns)

    def add_detection(self, frame_index, markers_data):
        """
        markers_data: lista tupli (id, x, y)
        frame_index: numer klatki
        """
        row_data = {'frame_index': frame_index}
        # Inicjalizacja wszystkich współrzędnych jako NaN
        for id in range(self.expected_markers):
            row_data[f'id{id}_x'] = np.nan
            row_data[f'id{id}_y'] = np.nan

        # Wypełnienie wykrytych markerów
        for marker_id, x, y in markers_data:
            if marker_id < self.expected_markers:
                row_data[f'id{marker_id}_x'] = x
                row_data[f'id{marker_id}_y'] = y

        self.df.loc[len(self.df)] = row_data