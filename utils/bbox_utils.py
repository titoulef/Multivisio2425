import numpy as np
import cv2
import random
import os
from personn import Personn, Population
from personn.Suitcase import Suitcase


def generate_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    cx=int((x1+x2)/2)
    cy=int(y2)
    return (cx, cy)

def get_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def draw_bboxes_stream(frame, track_id, bbox, color):
        x1, y1, x2, y2 = bbox
        cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

def bbox_distance(bbox1, bbox2):
    c1=get_center(bbox1)
    c2=get_center(bbox2)
    distance = get_distance(c1, c2)
    return distance

def bbox_covering(bbox1, bbox2, threshold=0.05, type='center'):
    if type == 'intersection':
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        delta = abs(x4-x3)*0.25  # augmentation de la bbox suitcase
        x3-=delta
        y3-=delta
        x4+=delta
        y4+=delta
        # Ensure areas are positive
        area_personn = abs(x2 - x1) * abs(y2 - y1)
        area_suitcase = abs(x4 - x3) * abs(y4 - y3)


        # Calculate the intersection rectangle
        xleft = max(x1, x3)
        xright = min(x2, x4)
        ytop = max(y1, y3)
        ybottom = min(y2, y4)
        if xright < xleft or ybottom < ytop:
            return False  # No intersection
        deltax = xright - xleft
        deltay = ybottom - ytop
        intersection = deltax * deltay
        # Check if the intersection covers enough of the smaller area
        if intersection > threshold * area_suitcase:
            return True
        else:
            return False

    elif type == 'center':
        distance = bbox_distance(bbox1, bbox2)
        if distance > 0.5 * (bbox1[3] - bbox1[1]):
            return False
        else:
            return True

def valisePersonne(frame, player_dict, suitcase_dict, population):
    lien_dict = {}
    drawn_bboxes = set()  # Ensemble pour suivre les bboxes déjà dessinées
    for track_id, data in player_dict.items():
        bbox1 = data['bbox']
        color = data['color']
        linked = False  # Flag pour savoir si on a associé un joueur à une valise

        for track_id2, bbox2 in suitcase_dict.items():
            if bbox_covering(bbox1, bbox2, type='center'):
                lien_dict[track_id2] = track_id
                # Dessin des bboxes si elles ne l'ont pas encore été
                if track_id not in drawn_bboxes:
                    #pers = Personn(track_id, bbox1, frame, color)
                    #population.addPerson(pers)

                    draw_bboxes_stream(frame, track_id, bbox1, color)
                    drawn_bboxes.add(track_id)
                if track_id2 not in drawn_bboxes:
                    draw_bboxes_stream(frame, track_id2, bbox2, color)
                    drawn_bboxes.add(track_id2)

                linked = True  # Un lien a été trouvé
                break  # On arrête la recherche pour ce joueur

        # Si aucun lien trouvé, dessiner en blanc
        if not linked:
            if track_id not in drawn_bboxes:
                #population.addPerson(Personn(track_id, bbox1, frame, color))
                draw_bboxes_stream(frame, track_id, bbox1, (255, 255, 255))
                drawn_bboxes.add(track_id)

    # Dessiner les valises non associées

    for track_id2, bbox2 in suitcase_dict.items():
        if track_id2 not in drawn_bboxes:
            draw_bboxes_stream(frame, track_id2, bbox2, (255, 255, 255))
            drawn_bboxes.add(track_id2)

    return lien_dict


def snapshop(frame, bbox, ID):
    # Extraire les coordonnées
    x1, y1, x2, y2 = map(int, bbox)

    # Vérifier si les coordonnées sont valides
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        raise ValueError("Les coordonnées bbox sont en dehors des limites de l'image.")

    # Vérifier/créer le dossier de sauvegarde
    output_dir = "ID_snapshots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extraire et sauvegarder le sous-image
    snapshot = frame[y1:y2, x1:x2]
    output_path = os.path.join(output_dir, f"ID{ID}.png")
    success = cv2.imwrite(output_path, snapshot)

    if not success:
        raise IOError(f"Échec de la sauvegarde de l'image dans {output_path}")

def get_the_closest_keypoint_index(point, keypoints, keypoint_indicees):
    closest_distance = float('inf')
    key_point_ind = keypoint_indicees[0]
    for keypoint_indix in keypoint_indicees:
        keypoint = keypoints[keypoint_indix*2], keypoints[keypoint_indix*2+1]
        distance =  get_distance(point, keypoint)

        if distance<closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_indix

    return key_point_ind

def get_height_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return y2-y1

def measure_xy_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return abs(x2-x1), abs(y2-y1)




