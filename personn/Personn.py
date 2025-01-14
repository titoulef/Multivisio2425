import utils
import cv2

class Personn():
    def __init__(self, ID, bbox, snapshop, suitcase=False):
        #self.pos2D = pos2D
        self.bbox=bbox
        self.ID = ID
        self.img = snapshop
        self.suitcase = []
        self.NbSuitcase = len(self.suitcase)
        self.suitcaseImg = None

    def __str__(self):
        return f"ID: {self.ID}, pos: {self.bbox}"

    def convertBboxToMapCoord(self):
        #les scotchs au sol 3*3m
        pos = utils.get_center(self.bbox)
        pos= utils.getXandY_from_a_origin(pos, (275, 205))
        p1= (275, 205)
        p2 =(545, 205)
        p3=(180, 320)
        utils.get_distance(p1, p2)
        xfin = pos[0]*3/utils.get_distance(p1, p2)
        yfin = pos[1]*3/utils.get_distance(p1, p3)
        return xfin, yfin

