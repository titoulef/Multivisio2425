import cv2
import numpy as np

import trackers
from utils import bbox_utils, get_distance
from mini_map import MiniMap, MapDetector
from utils.bbox_utils import bbox_covering, valisePersonne


def IdPersonn( input_video_path, fpsDivider, videoScale):
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
                print("Erreur lecture frame (show.py)")

            videoScale = float(videoScale)
            frame = cv2.resize(frame, None, fx=videoScale, fy=videoScale, interpolation=cv2.INTER_CUBIC)

            player_detection = player_tracker.detect_frames_stream(frame)
            suitcase_detection = suitcase_tracker.detect_frames_stream(frame)
            #interpolate the suitcase position
            #suitcase_detection = suitcase_tracker.interpolate_suitcase_position(suitcase_detection)
            #draw bounding boxes
            lien = valisePersonne(frame, player_detection, suitcase_detection)

            #draw mini mini_map
            #frame = MiniMap(frame).draw_mini_map(frame)

            #draw keypoints
            mapKey = MapDetector()
            mapKey.add_keypoint((150, 370))
            mapKey.add_keypoint((500, 410))
            mapKey.add_keypoint((600, 310))
            mapKey.draw_keypoints(frame)

            #draw lines between the player and the suitcase

            cv2.imshow("output", frame)
        cpt += 1
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cameraIP.release()
    cv2.destroyAllWindows()