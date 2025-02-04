import utils
import cv2

class Personn():
    def __init__(self, ID, bbox, frame, color):
        #self.pos2D = pos2D
        self.bbox=bbox
        self.ID = ID
        self.img = utils.snapshop(frame, bbox, ID)
        self.suitcase = False
        self.suitcase_ID = None
        self.color = color
        self.cpt = 0

    def __str__(self):
        return f"ID: {self.ID}, pos: {self.bbox}"

    def set_suitcase(self):
        self.suitcase = True


    def update(self,bbox):
        self.bbox = bbox
        if self.cpt == 10:
            self.cpt = 0




