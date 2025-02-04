import utils
import cv2

class Suitcase():
    def __init__(self, ID, bbox, suitcase=False):
        #self.pos2D = pos2D
        self.bbox = bbox
        self.ID = ID
        self.img =
        self.suitcase = []
        self.NbSuitcase = len(self.suitcase)
        self.suitcaseImg = None

    def __str__(self):
        return f"ID: {self.ID}, pos: {self.bbox}"



