import cv2
import sys

from fontTools.misc.bezierTools import segmentPointAtT

import constants
import utils
import numpy as np
from utils import get_center, get_distance, get_height_of_bbox, convert_pixels_to_meters
import personn

sys.path.append('../')

class MiniMap():
    def __init__(self, frame):
        # map placer en haut à droite par default, voir set_canvas_background_box_position
        self.draw_rect_width = 150
        self.draw_rect_height = 150
        self.buffer=10 #marge autour du background
        self.padding_map=5 #marge autour de map dans le backround

        self.set_canvas_background_box_position(frame)
        self.set_mini_map_position()
        self.set_map_drawing_key_points()
        self.set_map_lines()



        #self.pop=pop

    #set le grand backround rect
    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.draw_rect_height
        self.start_x = self.end_x - self.draw_rect_width
        self.start_y = self.end_y - self.draw_rect_height

    #set les keypoints
    def set_map_drawing_key_points(self):
        draw_key_points = [0]*8
        #haut gauche
        draw_key_points[0], draw_key_points[1] = int(self.map_start_x), int(self.map_start_y)
        #haut droite
        draw_key_points[2], draw_key_points[3] = int(self.map_end_x), int(self.map_start_y)
        #bas droite
        draw_key_points[4], draw_key_points[5] = int(self.map_end_x), int(self.map_end_y)
        #bas gauche
        draw_key_points[6], draw_key_points[7] = int(self.map_start_x), int(self.map_end_y)

        self.draw_key_points = draw_key_points

    #set les lignes
    def set_map_lines(self):
        self.lines = {
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0)
        }

    #set la la map
    def set_mini_map_position(self):
        self.map_start_x = self.start_x + self.padding_map
        self.map_start_y = self.start_y + self.padding_map
        self.map_end_x = self.end_x - self.padding_map
        self.map_end_y = self.end_y - self.padding_map
        self.map_drawing_width = self.map_end_x - self.map_start_x

    def get_mini_court_coordinates(self, object_position,
                                   closest_key_point,
                                   closest_key_point_index,
                                   player_height_in_pixels,
                                   player_height_in_meters
                                   ):
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = utils.measure_xy_distance(object_position, closest_key_point)

        #convertir en mètres
        distance_from_keypoint_x_meters = utils.convert_pixels_to_meters(distance_from_keypoint_x_pixels,
                                                                        player_height_in_meters,
                                                                        player_height_in_pixels
                                                                        )
        distance_from_keypoint_y_meters = utils.convert_pixels_to_meters(distance_from_keypoint_y_pixels,
                                                                         player_height_in_meters,
                                                                         player_height_in_pixels
                                                                         )
        #convert to mini map coordinates
        mini_court_x_distance_pixels = utils.convert_meters_to_pixel_distance(distance_from_keypoint_x_meters, constants.PLAYER_HEIGHT, player_height_in_pixels )
        mini_court_y_distance_pixels = utils.convert_meters_to_pixel_distance(distance_from_keypoint_y_meters, constants.PLAYER_HEIGHT, player_height_in_pixels )
        closest_mini_map_keypoint = (self.draw_key_points[closest_key_point_index*2],
                                     self.draw_key_points[closest_key_point_index*2+1])

        mini_court_player_position = (closest_mini_map_keypoint[0] + mini_court_x_distance_pixels,
                                        closest_mini_map_keypoint[1] + mini_court_y_distance_pixels)

        if mini_court_player_position[0] < self.map_start_x-self.padding_map:
            mini_court_player_position = (self.map_start_x- self.padding_map, mini_court_player_position[1])
        if mini_court_player_position[0] > self.map_end_x+self.padding_map:
            mini_court_player_position = (self.map_end_x+self.padding_map, mini_court_player_position[1])
        if mini_court_player_position[1] < self.map_start_y-self.padding_map:
            mini_court_player_position = (mini_court_player_position[0], self.map_start_y-self.padding_map)
        if mini_court_player_position[1] > self.map_end_y+self.padding_map:
            mini_court_player_position = (mini_court_player_position[0], self.map_end_y+self.padding_map)

        return mini_court_player_position

    def draw_map_key(self, frame):
        for i in range(0, len(self.draw_key_points), 2):
            x = int(self.draw_key_points[i])
            y = int(self.draw_key_points[i+1])
            cv2.circle(frame, (x,y), 2, (255, 0, 0), -1)
            cv2.putText(frame, f"KP: {i//2}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                        (255, 255, 255), 1)
        return frame

    def draw_map_lines(self, frame):
        for line in self.lines:
            start_point = (int(self.draw_key_points[line[0]*2]), int(self.draw_key_points[line[0]*2+1]))
            end_point = (int(self.draw_key_points[line[1]*2]), int(self.draw_key_points[line[1]*2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 1)

        return frame

    def draw_backround_rect(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        #draw rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha=0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out

    #draw la map final
    def draw_mini_map(self, frame):
        frame = self.draw_backround_rect(frame)
        frame = self.draw_map_lines(frame)
        frame = self.draw_map_key(frame)
        return frame

    #guetter
    def get_start_point_of_minimap(self):
        return (self.map_start_x, self.map_start_y)
    def get_width_of_minimap(self):
        return self.map_drawing_width
    def get_map_drawing_keypoints(self):
        return self.draw_key_points

    def get_the_closest_keypoint_index(self, foot_position):
        list=[]
        for kp in self.draw_key_points:
            list.append(get_distance(kp, foot_position))
        return np.min(list)

    """
    def convert_bounding_boxes_to_map_coordinates(self, player_boxes, suitcase_boxes, original_court_ket_points):
        player_heights = constants.PLAYER_HEIGHT

        output_suitcase_bboxes_dict = {}
        output_player_bboxes_dict = {}
        for player_id, data in player_boxes.items():
            bbox = data['bbox']
            color = data['color']
            foot_position = get_center(bbox)

            closest_key_point_index = utils.get_the_closest_keypoint_index(foot_position, original_court_ket_points, [0,1,2,3])
            closest_key_point = (original_court_ket_points[closest_key_point_index*2],
                                 original_court_ket_points[closest_key_point_index*2+1])
            print(player_id, closest_key_point_index)
            max_player_height_in_pixels = get_height_of_bbox(bbox)

            mini_map_player_position = self.get_mini_court_coordinates(foot_position,
                                                                        closest_key_point,
                                                                        closest_key_point_index,
                                                                        max_player_height_in_pixels,
                                                                        player_heights
                                                                        )
            output_player_bboxes_dict[player_id] = mini_map_player_position

        return output_player_bboxes_dict, output_suitcase_bboxes_dict
    """
    def convert_bounding_boxes_to_map_coordinates(self, frame, player_boxes, suitcase_boxes, keypoints):

        output_suitcase_bboxes_dict = {}
        output_player_bboxes_dict = {}
        for player_id, data in player_boxes.items():
            bbox = data['bbox']
            color = data['color']
            foot_position = get_center(bbox)

            ratioh, ratiov, = utils.get_axes_x_y_intersection_segments(frame, foot_position, keypoints, print=True)
            if ratioh is not None or ratiov is not None:
                output_player_bboxes_dict[player_id] = (int(self.map_start_x+ratioh*self.draw_rect_width), int(self.map_start_y+ratiov*self.draw_rect_height))
            else:
                output_player_bboxes_dict[player_id] = None
        return output_player_bboxes_dict, output_suitcase_bboxes_dict


    def draw_pints_on_mini_map(self, frame, pos, colors=(255, 0, 0)):
        for id, position in pos.items():
            if position is not None:
                x=int(position[0])
                y=int(position[1])
                cv2.circle(frame, (x, y), 2, colors, -1)
                cv2.putText(frame, f"ID: {id}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                            (255, 255, 255), 1)
        return frame