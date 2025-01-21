import numpy as np
import cv2

def convert_pixels_to_meters(pixels_dist, reference_height_in_meters, reference_height_in_pixels):
    return pixels_dist * reference_height_in_meters / reference_height_in_pixels

def convert_meters_to_pixel_distance(meters, reference_height_in_meters, reference_height_in_pixels):
    return meters * reference_height_in_pixels / reference_height_in_meters

def get_angle_from_x(segment):
    x, y = segment
    return np.arctan(y/x)

def get_angle_from_y(segment):
    x, y = segment
    return np.arctan(y/x)

def detect_intersection(vector_origin, vector_direction, segment_start, segment_end):
    """
    Détecte l'intersection entre un vecteur et un segment.

    :param vector_origin: Coordonnées d'origine du vecteur (x, y).
    :param vector_direction: Direction du vecteur (dx, dy).
    :param segment_start: Coordonnées du premier point du segment (x, y).
    :param segment_end: Coordonnées du second point du segment (x, y).
    :return: Coordonnées de l'intersection (x, y) ou None s'il n'y a pas d'intersection.
    """
    # Convertir les points en numpy arrays pour simplifier les calculs
    p = np.array(vector_origin)
    d = np.array(vector_direction)
    a = np.array(segment_start)
    b = np.array(segment_end)

    # Calcul des vecteurs pour le segment
    ab = b - a
    ap = p - a

    # Résolution du système linéaire
    matrix = np.array([-d, ab]).T  # Matrice des coefficients
    try:
        # Résoudre le système : t1 * d + t2 * ab = ap
        t, u = np.linalg.solve(matrix, ap)
    except np.linalg.LinAlgError:
        # Si la matrice est singulière, le vecteur est parallèle au segment
        return None

    # Vérifier les conditions pour une intersection
    if t >= 0 and 0 <= u <= 1:
        # Calculer le point d'intersection
        intersection = p + t * d
        return tuple(intersection)

    return None

def get_segs(keypoints):
    seg01 = (keypoints[2] - keypoints[0], keypoints[3] - keypoints[1])
    seg12 = (keypoints[4] - keypoints[2], keypoints[5] - keypoints[3])
    seg32 = (keypoints[6] - keypoints[4], keypoints[7] - keypoints[5])
    seg03 = (keypoints[6] - keypoints[0], keypoints[7] - keypoints[1])
    segs = [seg01, seg12, seg32, seg03]
    return segs

def norms(vect):
    x,y=vect
    return (x**2 + y**2)**0.5

def get_segs_norms(keypoints):
    up, right, down, left = get_segs(keypoints)
    up=norms(up)
    right=norms(right)
    down=norms(down)
    left=norms(left)
    return up, right, down, left

def get_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def get_axes_x_y(keypoints):
    segs=get_segs(keypoints)
    theta_down = get_angle_from_x(segs[2])
    theta_up = get_angle_from_x(segs[0])
    phi_left = get_angle_from_y(segs[3])
    phi_right = get_angle_from_y(segs[1])
    theta = np.mean([theta_down, theta_up])
    phi = np.mean([phi_left, phi_right])
    dirx = (np.cos(theta), np.sin(theta))
    diry = (np.sin(phi), np.cos(phi))
    return dirx, diry

def get_axes_x_y_intersection_segments(frame, foot_position, keypoints, print=True):
    dirx, diry = get_axes_x_y(keypoints)
    kp_hl=(keypoints[0], keypoints[1])
    kp_hr=(keypoints[2], keypoints[3])
    kp_br=(keypoints[4], keypoints[5])
    kp_bl=(keypoints[6], keypoints[7])
    interx_l = detect_intersection(foot_position, (-dirx[0], -dirx[1]), (keypoints[0], keypoints[1]),
                                         (keypoints[6], keypoints[7]))
    interx_r = detect_intersection(foot_position, dirx, (keypoints[2], keypoints[3]),
                                         (keypoints[4], keypoints[5]))

    intery_up = detect_intersection(foot_position, (-diry[0], -diry[1]), (keypoints[0], keypoints[1]),
                                          (keypoints[2], keypoints[3]))
    intery_dwn = detect_intersection(foot_position, diry, (keypoints[4], keypoints[5]),
                                           (keypoints[6], keypoints[7]))

    up, right, down, left = get_segs_norms(keypoints)
    ratioh = None
    ratiov = None
    if (interx_l and interx_r) and (intery_up and intery_dwn) is not None:
        if print:
            cv2.line(frame, (int(interx_l[0]), int(interx_l[1])), (int(interx_r[0]), int(interx_r[1])), (255, 255, 255),
                     1)
            cv2.line(frame, (int(intery_up[0]), int(intery_up[1])), (int(intery_dwn[0]), int(intery_dwn[1])),
                     (255, 255, 255), 1)
        intery_up=get_distance(kp_hl, intery_up)
        intery_dwn=get_distance(kp_bl, intery_dwn)
        interx_l=get_distance(kp_hl, interx_l)
        interx_r=get_distance(kp_hr, interx_r)

        ratioh = np.mean([intery_up / up, intery_dwn / down])
        ratiov = np.mean([interx_l / left, interx_r / right])

    return ratioh, ratiov
