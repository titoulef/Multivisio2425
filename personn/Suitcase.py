import utils
import cv2

class Suitcase():
    def __init__(self, ID, bbox):
        #self.pos2D = pos2D
        self.bbox = bbox
        self.ID = ID


    def __str__(self):
        return f"ID: {self.ID}, pos: {self.bbox}"







