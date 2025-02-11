from ultralytics import YOLO
import pandas as pd
import cv2
import numpy as np
from deep_sort.deep_sort import DeepSort
DEEP_SORT_WEIGHTS = 'C:/Ensta/Tracking/wetransfer_deep_sort_2025-02-04_1256/deep_sort/deep/checkpoint/ckpt.t7'


class SuitcaseTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = DeepSort(model_path=DEEP_SORT_WEIGHTS, max_age=100)

    def interpolate_suitcase_position(self, suitcase_positions):
        # Extraire les positions pour la clé 5 de chaque frame
        frames = list(suitcase_positions.keys())
        positions = [suitcase_positions[frame][5] if isinstance(suitcase_positions[frame], dict) else [np.nan, np.nan, np.nan, np.nan] for frame in frames]

        # Créer un DataFrame à partir des positions
        df_positions = pd.DataFrame(positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpoler les valeurs manquantes
        df_positions = df_positions.interpolate(method='linear')  # Interpolation linéaire
        df_positions = df_positions.bfill()  # Remplissage vers l'arrière
        df_positions = df_positions.fillna(0)  # Remplir tout reste avec 0

        # Reconstruire le dictionnaire avec les données interpolées
        for i, frame in enumerate(frames):
            suitcase_positions[frame][5] = df_positions.iloc[i].tolist()
        return suitcase_positions


    def detect_frameZ(self, frame):
        results = self.model.track(frame, persist=True, classes=[28], verbose=False)
        id_name_dict = results[0].names
        suitcase_dict = {}
        for box in results[0].boxes:
            track_id = int(box.id.tolist()[0]) if box.id is not None else -1
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]

            suitcase_dict[track_id] = result
        return suitcase_dict


    #deep_sort
    def detect_frame(self, frame):
        results = self.model(frame, classes=[28], conf=0.1, verbose=False)
        detections = []

        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = box
            w, h = int(x2 - x1), int(y2 - y1)
            x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            detections.append(([x, y, w, h], float(conf), int(cls_id)))

        if detections:
            bboxes_xywh = np.array([det[0] for det in detections], dtype=float)
            confidences = np.array([det[1] for det in detections], dtype=float)
            tracks = self.tracker.update(bboxes_xywh, confidences, frame)
            player_dict = {}

            for track in tracks:
                track_id = int(track[4])
                x1, y1, x2, y2 = map(int, track[:4])
                cls_id = track[5] if len(track) > 5 else -1
                player_dict[track_id] = [x1, y1, x2, y2]

        else:
            player_dict = {}

        return player_dict

