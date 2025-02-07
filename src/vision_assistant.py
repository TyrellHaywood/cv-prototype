from object_detection import ObjectDetector
from edge_detection import EdgeDetector
from tts import SpeechProcessor
import cv2
import time

class VisionAssistant:
    def __init__(self):
        self.detector = ObjectDetector()
        self.edge_detector = EdgeDetector()
        self.tts = SpeechProcessor()
        self.last_edge_alert = 0

    def process_frame(self, frame, show_bboxes=True, show_overlay=True, tts_enabled=True):
        """Handles frame processing: object detection, edge detection, and speech feedback."""
        current_time = time.time()

        # Object detection
        detected_objects = self.detector.detect_objects(frame, show_bboxes)
        self.tts.handle_announcements(detected_objects)

        # Edge detection & obstruction warnings
        edges = self.edge_detector.detect_edges(frame)
        self.edge_detector.detect_obstructions(frame, edges, current_time, self.tts, show_overlay)

        # Merge edges into frame
        frame = cv2.addWeighted(frame, 0.8, edges, 0.2, 0)
        return frame
