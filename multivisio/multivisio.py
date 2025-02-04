import cv2
import numpy as np

import trackers
from utils import bbox_utils, get_distance
from mini_map import MiniMap, MapDetector
from utils.bbox_utils import bbox_covering, valisePersonne
from personn import Population

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
    population = Population()
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
            #suitcase_detection_dict = suitcase_tracker.interpolate_suitcase_position(suitcase_detection_dict)
            mini_map = MiniMap(frame)
            map_detector = MapDetector()
            #sortir ca de la loop
            map_keypoints= map_detector.detect_keypoints(frame)
            map_detector.draw_keypoints(frame)


            #draw bounding boxes
            lien_dict = valisePersonne(frame, player_detection_dict, suitcase_detection_dict, population)

            # Convert bounding boxes to map coordinates
            player_mini_map_detections, suitcase_mini_map_detections = mini_map.convert_bounding_boxes_to_map_coordinates(
                frame, player_detection_dict, suitcase_detection_dict, map_keypoints, lien_dict, print=False)

            #print(pop)
            #draw mini mini_map convert_bounding_boxes_to_map_coordinates
            frame = mini_map.draw_mini_map(frame)
            frame = mini_map.draw_pints_on_mini_mapD(frame, player_mini_map_detections, what='pers')
            frame = mini_map.draw_pints_on_mini_mapD(frame, suitcase_mini_map_detections, what='suit')

            cv2.imshow("output", frame)
        cpt += 1
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cameraIP.release()
    cv2.destroyAllWindows()

def loop2(input_video_path1, input_video_path2, fpsDivider, videoScale):
    cameraIP1 = cv2.VideoCapture(input_video_path1)
    cameraIP2 = cv2.VideoCapture(input_video_path2)

    #'C:/Ensta/Tracking/best.pt'
    player_tracker1 = trackers.PlayerTracker(model_path='yolov10n')
    suitcase_tracker1 = trackers.SuitcaseTracker(model_path='yolov10n')
    player_tracker2 = trackers.PlayerTracker(model_path='yolov10n')
    suitcase_tracker2 = trackers.SuitcaseTracker(model_path='yolov10n')

    if not cameraIP1.isOpened():
        print("Erreur cam 1")
    if not cameraIP2.isOpened():
        print("Erreur cam 2")

    cv2.namedWindow("output1")
    cv2.namedWindow("output2")
    cv2.setMouseCallback("output1", mouse_callback)

    cpt = 0
    while True:
        ret1, frame1 = cameraIP1.read()
        ret2, frame2 = cameraIP2.read()
        if cpt % fpsDivider == 0:
            if not ret1:
                print("Erreur lecture cam1 frame (multivisio.py)")
            if not ret2:
                print("Erreur lecture cam2 frame (multivisio.py)")

            videoScale = float(videoScale)
            frame1 = cv2.resize(frame1, None, fx=videoScale, fy=videoScale, interpolation=cv2.INTER_CUBIC)
            frame2 = cv2.resize(frame2, None, fx=videoScale, fy=videoScale, interpolation=cv2.INTER_CUBIC)

            player_detection_dict1 = player_tracker1.detect_frame(frame1)
            suitcase_detection_dict1 = suitcase_tracker1.detect_frame(frame1)
            player_detection_dict2 = player_tracker2.detect_frame(frame2)
            suitcase_detection_dict2 = suitcase_tracker2.detect_frame(frame2)
            #interpolate the suitcase position
            #suitcase_detection_dict = suitcase_tracker.interpolate_suitcase_position(suitcase_detection_dict)
            mini_map = MiniMap(frame1)
            map_detector = MapDetector()
            #sortir ca de la loop
            map_keypoints= map_detector.detect_keypoints(frame1)
            map_detector.draw_keypoints(frame1)


            #draw bounding boxes
            pop1 = valisePersonne(frame1, player_detection_dict1, suitcase_detection_dict1)
            pop2 = valisePersonne(frame2, player_detection_dict2, suitcase_detection_dict2)


            # Convert bounding boxes to map coordinates
            player_mini_map_detections, suitcase_mini_map_detections = mini_map.convert_bounding_boxes_to_map_coordinates(
                frame1, player_detection_dict1, suitcase_detection_dict1, map_keypoints, print=True)

            #print(pop)
            #draw mini mini_map convert_bounding_boxes_to_map_coordinates
            frame1 = mini_map.draw_mini_map(frame1)
            frame1 = mini_map.draw_pints_on_mini_map(frame1, player_mini_map_detections)
            frame1 = mini_map.draw_pints_on_mini_map(frame1, suitcase_mini_map_detections, colors=(255, 255, 0))

            cv2.imshow("output1", frame1)
            cv2.imshow("output2", frame2)
        cpt += 1
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cameraIP1.release()
    cameraIP2.release()
    cv2.destroyAllWindows()