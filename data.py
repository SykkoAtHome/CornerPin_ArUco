import pandas as pd
import numpy as np


class Data:
    def __init__(self, expected_markers):
        self.expected_markers = expected_markers

        # Tworzenie kolumn dla DataFrame
        columns = ['frame_index']

        # Kolumny dla każdego markera
        for id in range(expected_markers):
            # Współrzędne narożników (4 punkty, każdy ma x,y)
            for corner in range(4):
                columns.extend([
                    f'id{id}_corner{corner}_x',
                    f'id{id}_corner{corner}_y'
                ])
            # Środek markera (dla kompatybilności wstecznej i szybkiego dostępu)
            columns.extend([f'id{id}_center_x', f'id{id}_center_y'])
            # Kąt orientacji markera (w stopniach)
            columns.append(f'id{id}_angle')
            # Parametry detekcji
            columns.extend([
                f'id{id}_win_size',
                f'id{id}_thresh_const',
                f'id{id}_min_perim',
                f'id{id}_approx_acc',
                f'id{id}_corner_dist'
            ])

        self.df = pd.DataFrame(columns=columns)

    def add_detection(self, frame_index, markers_data):
        """
        Dodaje detekcje do DataFrame.

        markers_data: lista krotek (id, corners, params), gdzie:
            - id: identyfikator markera
            - corners: tablica narożników (1,4,2)
            - params: słownik parametrów detekcji lub None dla domyślnych
        frame_index: numer klatki
        """
        row_data = {'frame_index': frame_index}

        # Inicjalizacja wszystkich wartości jako NaN
        for id in range(self.expected_markers):
            # Narożniki
            for corner in range(4):
                row_data[f'id{id}_corner{corner}_x'] = np.nan
                row_data[f'id{id}_corner{corner}_y'] = np.nan
            # Środek i orientacja
            row_data[f'id{id}_center_x'] = np.nan
            row_data[f'id{id}_center_y'] = np.nan
            row_data[f'id{id}_angle'] = np.nan
            # Parametry
            row_data[f'id{id}_win_size'] = np.nan
            row_data[f'id{id}_thresh_const'] = np.nan
            row_data[f'id{id}_min_perim'] = np.nan
            row_data[f'id{id}_approx_acc'] = np.nan
            row_data[f'id{id}_corner_dist'] = np.nan

        # Wypełnienie wykrytych markerów
        for marker_id, corners, params in markers_data:
            if marker_id < self.expected_markers:
                # corners ma kształt (1,4,2), więc bierzemy corners[0]
                for corner_idx, point in enumerate(corners[0]):
                    row_data[f'id{marker_id}_corner{corner_idx}_x'] = point[0]
                    row_data[f'id{marker_id}_corner{corner_idx}_y'] = point[1]

                # Oblicz i zapisz środek
                center = corners[0].mean(axis=0)
                row_data[f'id{marker_id}_center_x'] = center[0]
                row_data[f'id{marker_id}_center_y'] = center[1]

                # Oblicz i zapisz kąt orientacji
                vec = corners[0][1] - corners[0][0]  # wektor od punktu 0 do 1
                angle = np.degrees(np.arctan2(vec[1], vec[0]))
                row_data[f'id{marker_id}_angle'] = angle

                # Zapisz parametry detekcji
                if params is not None:
                    row_data[f'id{marker_id}_win_size'] = params.adaptiveThreshWinSizeMin
                    row_data[f'id{marker_id}_thresh_const'] = params.adaptiveThreshConstant
                    row_data[f'id{marker_id}_min_perim'] = params.minMarkerPerimeterRate
                    row_data[f'id{marker_id}_approx_acc'] = params.polygonalApproxAccuracyRate
                    row_data[f'id{marker_id}_corner_dist'] = params.minCornerDistanceRate

        self.df.loc[len(self.df)] = row_data

    def get_marker_corners(self, frame_idx, marker_id):
        """
        Zwraca narożniki konkretnego markera dla danej klatki w formacie (1,4,2).
        """
        row = self.df[self.df['frame_index'] == frame_idx]
        if row.empty:
            return None

        row = row.iloc[0]
        corners = np.zeros((4, 2))
        for i in range(4):
            x = row[f'id{marker_id}_corner{i}_x']
            y = row[f'id{marker_id}_corner{i}_y']
            corners[i, 0] = x
            corners[i, 1] = y

        if np.isnan(corners).any():
            return None

        # Przekształć do formatu (1,4,2) wymaganego przez OpenCV
        return corners.reshape(1, 4, 2)

    def get_marker_params(self, frame_idx, marker_id):
        """
        Zwraca parametry detekcji dla konkretnego markera.
        """
        row = self.df[self.df['frame_index'] == frame_idx]
        if row.empty:
            return None

        row = row.iloc[0]
        params = {
            'win_size': row[f'id{marker_id}_win_size'],
            'thresh_const': row[f'id{marker_id}_thresh_const'],
            'min_perim': row[f'id{marker_id}_min_perim'],
            'approx_acc': row[f'id{marker_id}_approx_acc'],
            'corner_dist': row[f'id{marker_id}_corner_dist']
        }
        return params if not np.isnan(list(params.values())).any() else None