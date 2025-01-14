import cv2
import numpy as np

import trackers
from utils import bbox_utils, get_distance
from mini_map import MiniMap, MapDetector
from utils.bbox_utils import bbox_covering, valisePersonne


def loop( input_video_path, fpsDivider, videoScale):
    cameraIP = cv2.VideoCapture(input_video_path)
    player_tracker = trackers.PlayerTracker(model_path='yolov10n')
    suitcase_tracker = trackers.SuitcaseTracker(model_path='yolov10n')

    if not cameraIP.isOpened():
        print("Erreur 1")

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
            #draw bounding boxes
            pop = valisePersonne(frame, player_detection_dict, suitcase_detection_dict)
            #print(pop)
            #draw mini mini_map
            frame = MiniMap(frame).draw_mini_map(frame)


            #draw keypoints
            mapKey = MapDetector()
            mapKey.add_keypoint((275, 205))
            mapKey.add_keypoint((545, 205))
            mapKey.add_keypoint((640, 335))
            mapKey.add_keypoint((180, 320))
            mapKey.draw_keypoints(frame)


            #draw lines between the player and the suitcase

            cv2.imshow("output", frame)
        cpt += 1
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cameraIP.release()
    cv2.destroyAllWindows()