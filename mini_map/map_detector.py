import cv2
class MapDetector():
    def __init__(self):
        self.keypoints = []

    def add_keypoint(self, point):
        self.keypoints.append(point)

    def draw_keypoints(self, frame):
        for i in range(0, len(self.keypoints), 2):
            point = (self.keypoints[i], self.keypoints[i + 1])
            cv2.circle(frame, point, 3, (255, 255, 255), -1)
            cv2.putText(frame, f"KP: {i // 2}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def detect_keypoints(self, frame):
        #self.keypoints = [275, 205, 545, 205, 640, 335, 180, 320] # hall2

        #self.keypoints = [282, 244, 543, 251, 641, 388, 159, 369] # hall1
        self.keypoints = [225, 252, 447, 184, 561, 265, 261, 390]  # hall3
        #self.keypoints = [211, 213, 549, 126, 805, 251, 281, 478]  # hall3 extended

        return self.keypoints
