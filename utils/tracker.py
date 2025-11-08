from ultralytics import YOLO
from collections import defaultdict

class ObjectTracker:
    def __init__(self):
        self.next_id = 0
        self.objects = defaultdict(dict)

    def update(self, detections):
        ids = []
        for det in detections:
            ids.append(self.next_id)
            self.next_id += 1
        return ids

