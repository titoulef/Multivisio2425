import utils
import cv2

class Personn():
    def __init__(self, ID, bbox, frame):
        #self.pos2D = pos2D
        self.bbox=bbox
        self.ID = ID
        self.img = utils.snapshop(frame, bbox, ID)
        self.suitcase = False
        self.suitcase_ID = None
        self.color = (255, 255, 255)

    def __str__(self):
        return f"ID: {self.ID}, pos: {self.bbox}"

    def set_suitcase(self, bbox_suitcase, suitcase_ID):
        self.suitcase = True
        self.suitcase_ID = suitcase_ID
        self.color = (255, 255, 0)




