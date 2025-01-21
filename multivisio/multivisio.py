import cv2
import numpy as np

import trackers
from utils import bbox_utils, get_distance
from mini_map import MiniMap, MapDetector
from utils.bbox_utils import bbox_covering, valisePersonne

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse position: ({x}, {y})")


def loop( input_video_path, fpsDivider, videoScale):
    cameraIP = cv2.VideoCapture(input_video_path)
    #'C:/Ensta/Tracking/best.pt'
    player_tracker = trackers.PlayerTracker(model_path='yolov10n')
    suitcase_tracker = trackers.SuitcaseTracker(model_path='yolov10n')

    if not cameraIP.isOpened():
        print("Erreur 1")

    cv2.namedWindow("output")
    cv2.setMouseCallback("output", mouse_callback)

    cpt = 0
    while True:
        ret, frame = cameraIP.read()
        if cpt % fpsDivider == 0:
            if not ret:
                print("Erreur lecture frame (multivisio.py)")

            videoScale = float(videoScale)
            frame = cv2.resize(frame, None, fx=videoScale, fy=videoScale, interpolation=cv2.INTER_CUBIC)

            player_detection_dict = player_tracker.detect_frame(frame)
            suitcase_detection_dict = suitcase_tracker.detect_frame(frame)
            #interpolate the suitcase position
            #suitcase_detection = suitcase_tracker.interpolate_suitcase_position(suitcase_detection)
            mini_map = MiniMap(frame)
            map_detector = MapDetector()
            #sortir ca de la loop
            map_keypoints= map_detector.detect_keypoints(frame)
            map_detector.draw_keypoints(frame)

            # Convert bounding boxes to map coordinates
            player_mini_map_detections, suitcase_mini_map_detections = mini_map.convert_bounding_boxes_to_map_coordinates(frame, player_detection_dict, suitcase_detection_dict, map_keypoints)

            #draw bounding boxes
            pop = valisePersonne(frame, player_detection_dict, suitcase_detection_dict)
            #print(pop)
            #draw mini mini_map convert_bounding_boxes_to_map_coordinates
            frame = mini_map.draw_mini_map(frame)
            frame = mini_map.draw_pints_on_mini_map(frame, player_mini_map_detections)


            #draw lines between the player and the suitcase

            cv2.imshow("output", frame)
        cpt += 1
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cameraIP.release()
    cv2.destroyAllWindows()